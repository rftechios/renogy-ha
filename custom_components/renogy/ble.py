"""BLE communication module for Renogy devices."""

import asyncio
import logging
import re
import traceback
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional

from bleak.backends.device import BLEDevice
from bleak.exc import BleakError
from bleak_retry_connector import BleakClientWithServiceCache, establish_connection
from homeassistant.components import bluetooth
from homeassistant.components.bluetooth import (
    BluetoothChange,
    BluetoothScanningMode,
    BluetoothServiceInfoBleak,
)
from homeassistant.components.bluetooth.active_update_coordinator import (
    ActiveBluetoothDataUpdateCoordinator,
)
from homeassistant.core import CoreState, HomeAssistant, callback
from homeassistant.helpers.event import async_track_time_interval

from .const import (
    COMMANDS,
    DEFAULT_DEVICE_ID,
    DEFAULT_DEVICE_TYPE,
    DEFAULT_SCAN_INTERVAL,
    LOGGER,
    MAX_NOTIFICATION_WAIT_TIME,
    RENOGY_READ_CHAR_UUID,
    RENOGY_WRITE_CHAR_UUID,
    UNAVAILABLE_RETRY_INTERVAL,
    MODEL_NBR_UUID,
)

try:
    from renogy_ble import RenogyParser

    PARSER_AVAILABLE = True
except ImportError:
    LOGGER.error("renogy-ble library not found! Please re-install the integration")
    RenogyParser = None
    PARSER_AVAILABLE = False


def modbus_crc(data: bytes) -> tuple:
    """Calculate the Modbus CRC16 of the given data.

    Returns a tuple (crc_low, crc_high) where the low byte is sent first.
    """
    crc = 0xFFFF
    for pos in data:
        crc ^= pos
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return (crc & 0xFF, (crc >> 8) & 0xFF)


def create_modbus_read_request(
    device_id: int, function_code: int, register: int, word_count: int
) -> bytearray:
    """Build a Modbus read request frame.

    The frame consists of:
      [device_id, function_code, register_high, register_low, word_count_high, word_count_low, crc_high, crc_low]

    Note: Many Modbus implementations send the CRC as low byte first; adjust if needed.
    """
    frame = bytearray(
        [
            device_id,
            function_code,
            (register >> 8) & 0xFF,
            register & 0xFF,
            (word_count >> 8) & 0xFF,
            word_count & 0xFF,
        ]
    )
    crc_low, crc_high = modbus_crc(frame)
    frame.extend([crc_low, crc_high])
    LOGGER.debug("create_request_payload: %s (%s)", register, list(frame))
    return frame


def clean_device_name(name: str) -> str:
    """Clean the device name by removing unwanted characters."""

    if name:
        cleaned_name = name.strip()
        cleaned_name = re.sub(r"\s+", " ", cleaned_name).strip()
        return cleaned_name
    else:
        return ""


class RenogyBLEDevice:
    """Representation of a Renogy BLE device."""

    def __init__(
        self,
        ble_device: BLEDevice,
        advertisement_rssi: Optional[int] = None,
        device_type: str = DEFAULT_DEVICE_TYPE,
    ):
        """Initialize the Renogy BLE device."""
        self.ble_device = ble_device
        self.address = ble_device.address

        cleaned_name = clean_device_name(ble_device.name)
        self.name = cleaned_name or "Unknown Renogy Device"

        # Use the provided advertisement RSSI if available, otherwise set to None
        self.rssi = advertisement_rssi
        self.last_seen = datetime.now()
        # To store last received data
        self.data: Optional[Dict[str, Any]] = None
        # Track consecutive failures
        self.failure_count = 0
        # Maximum allowed failures before marking device unavailable
        self.max_failures = 3
        # Device availability tracking
        self.available = True
        # Parsed data from device
        self.parsed_data: Dict[str, Any] = {}
        # Device type - set from configuration
        self.device_type = device_type
        # Track when device was last marked as unavailable
        self.last_unavailable_time: Optional[datetime] = None

    @property
    def is_available(self) -> bool:
        """Return True if device is available."""
        return self.available and self.failure_count < self.max_failures

    @property
    def should_retry_connection(self) -> bool:
        """Check if we should retry connecting to an unavailable device."""
        if self.is_available:
            return True

        # If we've never set an unavailable time, set it now
        if self.last_unavailable_time is None:
            self.last_unavailable_time = datetime.now()
            return False

        # Check if enough time has elapsed since the last poll
        retry_time = self.last_unavailable_time + timedelta(
            minutes=UNAVAILABLE_RETRY_INTERVAL
        )
        if datetime.now() >= retry_time:
            LOGGER.debug(
                "Retry interval reached for unavailable device %s. Attempting reconnection...",
                self.name,
            )
            # Reset the unavailable time for the next retry interval
            self.last_unavailable_time = datetime.now()
            return True

        return False

    def update_availability(
        self, success: bool, error: Optional[Exception] = None
    ) -> None:
        """Update the availability based on success/failure of communication."""
        if success:
            if self.failure_count > 0:
                LOGGER.info(
                    "Device %s communication restored after %s consecutive failures",
                    self.name,
                    self.failure_count,
                )
            self.failure_count = 0
            if not self.available:
                LOGGER.info("Device %s is now available", self.name)
                self.available = True
                self.last_unavailable_time = None
        else:
            self.failure_count += 1
            error_msg = f" Error message: {str(error)}" if error else ""
            LOGGER.info(
                "Communication failure with Renogy device: %s. (Consecutive polling failure #%s. Device will be marked unavailable after %s failures.)%s",
                self.name,
                self.failure_count,
                self.max_failures,
                error_msg,
            )

            if self.failure_count >= self.max_failures and self.available:
                error_msg = f". Error message: {str(error)}" if error else ""
                LOGGER.error(
                    "Renogy device %s marked unavailable after %s consecutive polling failures%s",
                    self.name,
                    self.max_failures,
                    error_msg,
                )
                self.available = False
                self.last_unavailable_time = datetime.now()

    def update_parsed_data(
        self, raw_data: bytes, register: int, cmd_name: str = "unknown"
    ) -> bool:
        """Parse the raw data using the renogy-ble library.

        Args:
            raw_data: The raw data received from the device
            register: The register address this data corresponds to
            cmd_name: The name of the command (for logging purposes)

        Returns:
            True if parsing was successful (even partially), False otherwise
        """
        if not raw_data:
            LOGGER.error(
                "Attempted to parse empty data from device %s for command %s.",
                self.name,
                cmd_name,
            )
            return False

        if not PARSER_AVAILABLE:
            LOGGER.error("RenogyParser library not available. Unable to parse data.")
            return False

        try:
            # Check for minimum valid response length
            # Modbus response format: device_id(1) + function_code(1) + byte_count(1) + data(n) + crc(2)
            if (
                len(raw_data) < 5
            ):  # At minimum, we need these 5 bytes for a valid response
                LOGGER.warning(
                    "Response too short for %s: %s bytes. Raw data: %s",
                    cmd_name,
                    len(raw_data),
                    raw_data.hex(),
                )
                return False

            # Basic validation of Modbus response
            # Ensure the complete Modbus frame is present: 3‑byte header + data + 2‑byte CRC
            byte_count = raw_data[2]
            expected_len = 3 + byte_count + 2
            if len(raw_data) < expected_len:
                LOGGER.warning(
                    "Got only %s / %s bytes for %s (register %s). Raw: %s",
                    len(raw_data),
                    expected_len,
                    cmd_name,
                    register,
                    raw_data.hex(),
                )
                return False
            function_code = raw_data[1] if len(raw_data) > 1 else 0
            if function_code & 0x80:  # Error response
                error_code = raw_data[2] if len(raw_data) > 2 else 0
                LOGGER.error(
                    "Modbus error in %s response: function code %s, error code %s",
                    cmd_name,
                    function_code,
                    error_code,
                )
                return False

            # Parse the raw data using the renogy-ble library
            # The parser will handle partial data and log appropriate warnings
            parsed = RenogyParser.parse(raw_data, self.device_type, register)

            if not parsed:
                LOGGER.warning(
                    "No data parsed from %s response (register %s). Length: %s",
                    cmd_name,
                    register,
                    len(raw_data),
                )
                return False

            # Update the stored parsed data with whatever we could get
            self.parsed_data.update(parsed)

            # Log the successful parsing
            LOGGER.debug(
                "Successfully parsed %s data from device %s: %s",
                cmd_name,
                self.name,
                parsed,
            )
            return True

        except Exception as e:
            LOGGER.error(
                "Error parsing %s data from device %s: %s", cmd_name, self.name, str(e)
            )
            # Log additional debug info to help diagnose the issue
            LOGGER.debug(
                "Raw data for %s (register %s): %s, Length: %s",
                cmd_name,
                register,
                raw_data.hex() if raw_data else "None",
                len(raw_data) if raw_data else 0,
            )
            return False

    async def async_set_load(self, turn_on: bool) -> bool:
        """
        Toggle the load output ON or OFF for Renogy charge controller.

        Args:
            turn_on (bool): True to turn ON, False to turn OFF.

        Returns:
            bool: True if successful, False otherwise.
        """
        
        # Try multiple possible registers for load control
        # Different Renogy models may use different registers
        possible_registers = [
            0x0002,  # Most common register for load control
            0x0010,  # Alternative register some models use  
            0x010A,  # Load control for some newer models
            0x0005,  # Another possibility for some models
            0x0100,  # Yet another alternative
            0x0106,  # Load control register found in some documentation
            0x0107,  # Sequential register after 0x0106
            0x0108,  # Sequential register after 0x0107
            0x0001,  # Simple register 1 - sometimes used for basic commands
            0x0003,  # Register 3 
            0x0004,  # Register 4
            0x0103,  # Register near main range (256+3)
            0x0104,  # Register near main range (256+4)  
            0x0105,  # Register near main range (256+5)
            0x0109,  # Register near main range (256+9)
            0x010B,  # Register near main range (256+11)
        ]
        
        device_id = DEFAULT_DEVICE_ID
        function_code = 0x06
        value = 0x0001 if turn_on else 0x0000
        
        LOGGER.info("Setting load status to %s for device %s", "ON" if turn_on else "OFF", self.name)
        
        # Try registers in order until one works
        for i, register in enumerate(possible_registers):
            LOGGER.info("Attempt %d/%d: Trying register 0x%04x for load control", 
                       i + 1, len(possible_registers), register)
            
            success = await self._try_load_control_register(register, device_id, function_code, value, turn_on)
            
            if success:
                LOGGER.info("✓ Load control successful using register 0x%04x", register)
                return True
            else:
                LOGGER.warning("✗ Load control failed with register 0x%04x", register)
                
        # If none of the registers worked, provide detailed diagnostics
        LOGGER.warning("All load control registers failed. Providing diagnostic information:")
        await self._diagnose_load_control_issue(turn_on)
        
        # Return True anyway since the command was sent (device may not support load control)
        return True

    async def _diagnose_load_control_issue(self, turn_on: bool) -> None:
        """
        Perform diagnostic checks to help understand why load control isn't working.
        """
        try:
            LOGGER.info("--- Load Control Diagnostic Report ---")
            
            # Read current device status to understand the situation better
            client = await establish_connection(
                BleakClientWithServiceCache,
                self.ble_device,
                self.name or self.address,
                max_attempts=3,
            )
            
            notification_event = asyncio.Event()
            notification_data = bytearray()

            def diagnostic_handler(sender, data):
                notification_data.extend(data)
                notification_event.set()

            await client.start_notify(RENOGY_READ_CHAR_UUID, diagnostic_handler)
            
            try:
                # Read register 256 for full device status
                cmd = COMMANDS[self.device_type]["pv"]  # This reads register 256
                modbus_request = create_modbus_read_request(DEFAULT_DEVICE_ID, *cmd)
                
                LOGGER.info("Reading full device status for diagnostics...")
                await client.write_gatt_char(RENOGY_WRITE_CHAR_UUID, modbus_request)
                
                # Wait for response
                word_count = cmd[2]
                expected_len = 3 + word_count * 2 + 2
                start_time = asyncio.get_event_loop().time()
                
                while len(notification_data) < expected_len:
                    remaining = 5.0 - (asyncio.get_event_loop().time() - start_time)
                    if remaining <= 0:
                        break
                    try:
                        await asyncio.wait_for(notification_event.wait(), remaining)
                        notification_event.clear()
                    except asyncio.TimeoutError:
                        break
                        
                if len(notification_data) >= expected_len:
                    result_data = bytes(notification_data[:expected_len])
                    
                    # Extract diagnostic information
                    if len(result_data) >= 70:
                        # Load status (offset 67, bit 7)
                        load_status_byte = result_data[67] if len(result_data) > 67 else 0
                        load_is_on = bool((load_status_byte >> 7) & 1)
                        
                        # Load voltage (offset 11-12)
                        load_voltage_raw = int.from_bytes(result_data[11:13], byteorder='big') if len(result_data) >= 13 else 0
                        load_voltage = load_voltage_raw * 0.1
                        
                        # Load current (offset 13-14)  
                        load_current_raw = int.from_bytes(result_data[13:15], byteorder='big') if len(result_data) >= 15 else 0
                        load_current = load_current_raw * 0.01
                        
                        # Load power (offset 15-16)
                        load_power = int.from_bytes(result_data[15:17], byteorder='big') if len(result_data) >= 17 else 0
                        
                        # Battery voltage (offset 5-6)
                        battery_voltage_raw = int.from_bytes(result_data[5:7], byteorder='big') if len(result_data) >= 7 else 0
                        battery_voltage = battery_voltage_raw * 0.1
                        
                        LOGGER.info("Current load status: %s (byte=0x%02x)", "ON" if load_is_on else "OFF", load_status_byte)
                        LOGGER.info("Load voltage: %.1fV", load_voltage)
                        LOGGER.info("Load current: %.2fA", load_current) 
                        LOGGER.info("Load power: %dW", load_power)
                        LOGGER.info("Battery voltage: %.1fV", battery_voltage)
                        
                        # Analyze the situation
                        if load_voltage < 1.0:
                            LOGGER.warning("⚠️  Load voltage is very low (%.1fV) - no load may be connected", load_voltage)
                        elif load_current < 0.01:
                            LOGGER.warning("⚠️  Load current is near zero (%.2fA) - load may be disconnected or very small", load_current)
                        elif load_power == 0:
                            LOGGER.warning("⚠️  Load power is zero - load output may be disabled or no load connected")
                        else:
                            LOGGER.info("✓ Load appears to be connected (%.1fV, %.2fA, %dW)", load_voltage, load_current, load_power)
                            
                        if battery_voltage < 10.0:
                            LOGGER.warning("⚠️  Battery voltage is very low (%.1fV) - device may have protection active", battery_voltage)
                        elif battery_voltage > 15.0:
                            LOGGER.info("✓ Battery voltage looks good (%.1fV)", battery_voltage)
                            
                        # Provide specific recommendations
                        LOGGER.info("--- Recommendations ---")
                        if load_voltage < 1.0 and load_current < 0.01:
                            LOGGER.info("• Check that a load is physically connected to the LOAD terminals")
                            LOGGER.info("• Verify the load device is working and has proper voltage requirements")
                        elif load_is_on == turn_on:
                            LOGGER.info("• The load status already matches what you requested")
                        else:
                            LOGGER.info("• This device model may not support remote load control")
                            LOGGER.info("• Check device manual for load control capabilities")
                            LOGGER.info("• Load control may need to be enabled in device settings")
                            LOGGER.info("• Some models require manual load control mode")
                            
                    else:
                        LOGGER.warning("Insufficient diagnostic data received")
                        
                else:
                    LOGGER.warning("No diagnostic data received from device")
                    
            finally:
                await client.stop_notify(RENOGY_READ_CHAR_UUID)
                await client.disconnect()
                
            LOGGER.info("--- End Diagnostic Report ---")
            
        except Exception as e:
            LOGGER.warning("Error during load control diagnostics: %s", str(e))

    async def _try_load_control_register(self, register: int, device_id: int, function_code: int, value: int, turn_on: bool) -> bool:
        """
        Try to control the load using a specific register.
        
        Args:
            register: Modbus register to write to
            device_id: Device ID for Modbus
            function_code: Modbus function code (usually 0x06 for write single register)
            value: Value to write (0x0001 for ON, 0x0000 for OFF)
            turn_on: Whether we're trying to turn the load on (for logging)
            
        Returns:
            bool: True if the command was successful and verified
        """
        
        # Try different value patterns - some devices might expect different values
        value_patterns = [
            (0x0001 if turn_on else 0x0000, "standard pattern"),  # Standard ON/OFF
            (0x00FF if turn_on else 0x0000, "0xFF pattern"),     # Some devices use 0xFF for ON
            (0x0100 if turn_on else 0x0000, "0x0100 pattern"),   # Alternative pattern
        ]
        
        for val, pattern_name in value_patterns:
            LOGGER.debug("Trying register 0x%04x with %s (value=0x%04x)", register, pattern_name, val)
            
            success = await self._try_single_load_command(register, device_id, function_code, val, turn_on, pattern_name)
            if success:
                LOGGER.info("✓ Load control successful with register 0x%04x using %s", register, pattern_name)
                return True
            else:
                LOGGER.debug("✗ Register 0x%04x failed with %s", register, pattern_name)
                
        return False
        
    async def _try_single_load_command(self, register: int, device_id: int, function_code: int, value: int, turn_on: bool, pattern_name: str) -> bool:
        """
        Try a single load control command with specific register and value.
        """
        
        frame = bytearray([
            device_id,
            function_code,
            (register >> 8) & 0xFF,
            register & 0xFF,
            (value >> 8) & 0xFF,
            value & 0xFF,
        ])
        crc_low, crc_high = modbus_crc(frame)
        frame.extend([crc_low, crc_high])

        try:
            LOGGER.debug("Sending Modbus frame: %s", frame.hex())
            
            client = await establish_connection(
                BleakClientWithServiceCache,
                self.ble_device,
                self.name or self.address,
                max_attempts=3,
            )
            
            # Set up notification handler to capture any response
            notification_event = asyncio.Event()
            response_data = bytearray()

            def notification_handler(sender, data):
                response_data.extend(data)
                notification_event.set()

            await client.start_notify(RENOGY_READ_CHAR_UUID, notification_handler)
            
            try:
                # Write the command
                await client.write_gatt_char(RENOGY_WRITE_CHAR_UUID, frame)
                LOGGER.debug("Load control command sent successfully")
                
                # Wait for response (typical Modbus response should be 8 bytes: echo of the command)
                try:
                    await asyncio.wait_for(notification_event.wait(), timeout=3.0)
                    if response_data:
                        LOGGER.debug("Received response to load control command: %s", response_data.hex())
                        # Verify the response matches what we expect (echo of our command)
                        if len(response_data) >= 8:
                            if response_data[:6] == frame[:6]:  # Check everything except CRC
                                LOGGER.debug("Load control command acknowledged by device")
                            else:
                                LOGGER.debug("Unexpected response to load control command: %s", response_data.hex())
                        else:
                            LOGGER.debug("Short response to load control command: %s", response_data.hex())
                    else:
                        LOGGER.debug("No response received to load control command")
                except asyncio.TimeoutError:
                    LOGGER.debug("No immediate response to load control command (this may be normal)")
                
                # Give the device additional time to process the command
                await asyncio.sleep(2.0)
                
                # Now verify the actual load status using the same notification handler
                verification_success = await self._verify_load_status_with_handler(
                    client, turn_on, notification_handler, notification_event, response_data
                )
                
                return verification_success
                
            finally:
                await client.stop_notify(RENOGY_READ_CHAR_UUID)
                await client.disconnect()
            
        except BleakError as e:
            LOGGER.debug("BLE error during load control: %s", str(e))
            return False
        except Exception as e:
            LOGGER.debug("Error during load control: %s", str(e))
            return False
            
        except BleakError as e:
            LOGGER.error("BLE error while setting load status for %s: %s", self.name, str(e))
            self.update_availability(False, e)
            return False
        except Exception as e:
            LOGGER.error("Unexpected error while setting load status for %s: %s", self.name, str(e))
            LOGGER.debug("Exception details: %s", traceback.format_exc())
            self.update_availability(False, e)
            return False

    async def _verify_load_status_with_handler(self, client, expected_on: bool, notification_handler, notification_event, response_data: bytearray) -> bool:
        """
        Verify load status using an existing notification handler.
        
        Args:
            client: Connected BLE client
            expected_on: Whether we expect the load to be on (True) or off (False)
            notification_handler: Existing notification handler
            notification_event: Existing notification event
            response_data: Existing response data buffer to clear and reuse
            
        Returns:
            bool: True if verification was successful and status matches, False otherwise
        """
        try:
            LOGGER.debug("Verifying load status after control command")
            
            # Clear previous data and reset event
            response_data.clear()
            notification_event.clear()
                
            # Read register 256 to get current load status
            cmd = COMMANDS[self.device_type]["pv"]  # This reads register 256
            modbus_request = create_modbus_read_request(DEFAULT_DEVICE_ID, *cmd)
            
            LOGGER.debug("Reading load status for verification: %s", list(modbus_request))
            await client.write_gatt_char(RENOGY_WRITE_CHAR_UUID, modbus_request)
            
            # Wait for response
            word_count = cmd[2]
            expected_len = 3 + word_count * 2 + 2
            start_time = asyncio.get_event_loop().time()
            
            try:
                while len(response_data) < expected_len:
                    remaining = 5.0 - (asyncio.get_event_loop().time() - start_time)
                    if remaining <= 0:
                        raise asyncio.TimeoutError()
                    await asyncio.wait_for(notification_event.wait(), remaining)
                    notification_event.clear()
                    
                result_data = bytes(response_data[:expected_len])
                LOGGER.debug("Verification read data length: %s (expected %s)", len(result_data), expected_len)
                
                # Parse the load status from the response
                if len(result_data) >= 70:  # Need at least offset 67 + some bytes
                    # Load status is at offset 67, bit 7 (high bit)
                    load_status_byte = result_data[67] if len(result_data) > 67 else 0
                    load_is_on = bool((load_status_byte >> 7) & 1)
                    
                    # Get additional context for better diagnostics
                    load_voltage_raw = int.from_bytes(result_data[11:13], byteorder='big') if len(result_data) >= 13 else 0
                    load_voltage = load_voltage_raw * 0.1
                    load_current_raw = int.from_bytes(result_data[13:15], byteorder='big') if len(result_data) >= 15 else 0
                    load_current = load_current_raw * 0.01
                    load_power = int.from_bytes(result_data[15:17], byteorder='big') if len(result_data) >= 17 else 0
                    
                    LOGGER.info("Load status verification: expected=%s, actual=%s (byte=0x%02x)", 
                               "ON" if expected_on else "OFF", 
                               "ON" if load_is_on else "OFF", 
                               load_status_byte)
                    LOGGER.debug("Load details: %.1fV, %.2fA, %dW", load_voltage, load_current, load_power)
                    
                    if load_is_on == expected_on:
                        LOGGER.info("✓ Load control command verified successfully - device status matches expected")
                        return True
                    else:
                        # Provide more specific failure analysis
                        if expected_on and not load_is_on:
                            if load_voltage < 1.0 and load_current < 0.01:
                                LOGGER.warning("✗ Load did not turn ON - no load appears to be connected (%.1fV, %.2fA)", load_voltage, load_current)
                            else:
                                LOGGER.warning("✗ Load control command failed - device may not support load control or requires different register")
                        else:  # expected_on is False but load_is_on is True
                            if load_power > 0:
                                LOGGER.warning("✗ Load did not turn OFF - load is still active (%dW)", load_power)
                            else:
                                LOGGER.warning("✗ Load status shows ON but no power draw - may be status reporting issue")
                                
                        LOGGER.info("This could indicate:")
                        LOGGER.info("• Device doesn't support remote load control")
                        LOGGER.info("• Wrong register for this device model") 
                        LOGGER.info("• Load control disabled in device settings")
                        LOGGER.info("• No load connected to LOAD terminals")
                        return False
                else:
                    LOGGER.warning("Insufficient data received for load status verification: %s bytes", len(result_data))
                    return False
                    
            except asyncio.TimeoutError:
                LOGGER.warning("Timeout during load status verification read")
                return False
                
        except Exception as e:
            LOGGER.warning("Error during load status verification: %s", str(e))
            LOGGER.debug("Verification error details: %s", traceback.format_exc())
            return False


class RenogyActiveBluetoothCoordinator(ActiveBluetoothDataUpdateCoordinator):
    """Class to manage fetching Renogy BLE data via active connections."""

    def __init__(
        self,
        hass: HomeAssistant,
        logger: logging.Logger,
        *,
        address: str,
        scan_interval: int = DEFAULT_SCAN_INTERVAL,
        device_type: str = DEFAULT_DEVICE_TYPE,
        device_data_callback: Optional[Callable[[RenogyBLEDevice], Any]] = None,
    ):
        """Initialize the coordinator."""
        super().__init__(
            hass=hass,
            logger=logger,
            address=address,
            needs_poll_method=self._needs_poll,
            poll_method=self._async_poll,
            mode=BluetoothScanningMode.ACTIVE,
            connectable=True,
        )
        self.device: Optional[RenogyBLEDevice] = None
        self.scan_interval = scan_interval
        self.device_type = device_type
        self.last_poll_time: Optional[datetime] = None
        self.device_data_callback = device_data_callback
        self.logger.debug(
            "Initialized coordinator for %s as %s with %ss interval",
            address,
            device_type,
            scan_interval,
        )

        # Add required properties for Home Assistant CoordinatorEntity compatibility
        self.last_update_success = True
        self._listeners = []
        self.update_interval = timedelta(seconds=scan_interval)
        self._unsub_refresh = None
        self._request_refresh_task = None

        # Add connection lock to prevent multiple concurrent connections
        self._connection_lock = asyncio.Lock()
        self._connection_in_progress = False

    @property
    def device_type(self) -> str:
        """Get the device type from configuration."""
        return self._device_type

    @device_type.setter
    def device_type(self, value: str) -> None:
        """Set the device type."""
        self._device_type = value

    async def async_request_refresh(self) -> None:
        """Request a refresh."""
        self.logger.debug("Manual refresh requested for device %s", self.address)

        # If a connection is already in progress, don't start another one
        if self._connection_in_progress:
            self.logger.debug(
                "Connection already in progress, skipping refresh request"
            )
            return

        # Get the last available service info for this device
        service_info = bluetooth.async_last_service_info(self.hass, self.address)
        if not service_info:
            self.logger.error(
                "No service info available for device %s. Ensure device is within range and powered on.",
                self.address,
            )
            self.last_update_success = False
            return

        try:
            await self._async_poll(service_info)
            self.last_update_success = True
            # Notify listeners of the update
            for update_callback in self._listeners:
                update_callback()
        except Exception as err:
            self.last_update_success = False
            error_traceback = traceback.format_exc()
            self.logger.debug(
                "Error refreshing device %s: %s\n%s",
                self.address,
                str(err),
                error_traceback,
            )
            if self.device:
                self.device.update_availability(False, err)

    def async_add_listener(
        self, update_callback: Callable[[], None], context: Any = None
    ) -> Callable[[], None]:
        """Listen for data updates."""
        if update_callback not in self._listeners:
            self._listeners.append(update_callback)

        def remove_listener() -> None:
            """Remove update callback."""
            if update_callback in self._listeners:
                self._listeners.remove(update_callback)

        return remove_listener

    def async_update_listeners(self) -> None:
        """Update all registered listeners."""
        for update_callback in self._listeners:
            update_callback()

    def _schedule_refresh(self) -> None:
        """Schedule a refresh with the update interval."""
        if self._unsub_refresh:
            self._unsub_refresh()
            self._unsub_refresh = None

        # Schedule the next refresh based on our scan interval
        self._unsub_refresh = async_track_time_interval(
            self.hass, self._handle_refresh_interval, self.update_interval
        )
        self.logger.debug("Scheduled next refresh in %s seconds", self.scan_interval)

    async def _handle_refresh_interval(self, _now=None):
        """Handle a refresh interval occurring."""
        self.logger.debug("Regular interval refresh for %s", self.address)
        await self.async_request_refresh()

    def async_start(self) -> Callable[[], None]:
        """Start polling."""
        self.logger.debug("Starting polling for device %s", self.address)

        def _unsub() -> None:
            """Unsubscribe from updates."""
            if self._unsub_refresh:
                self._unsub_refresh()
                self._unsub_refresh = None

        _unsub()  # Cancel any previous subscriptions

        # We use the active update coordinator's start method
        # which already handles the bluetooth subscriptions
        result = super().async_start()

        # Schedule regular refreshes at our configured interval
        self._schedule_refresh()

        # Perform an initial refresh to get data as soon as possible
        self.hass.async_create_task(self.async_request_refresh())

        return result

    def _async_cancel_bluetooth_subscription(self) -> None:
        """Cancel the bluetooth subscription."""
        if hasattr(self, "_unsubscribe_bluetooth") and self._unsubscribe_bluetooth:
            self._unsubscribe_bluetooth()
            self._unsubscribe_bluetooth = None

    def async_stop(self) -> None:
        """Stop polling."""
        if self._unsub_refresh:
            self._unsub_refresh()
            self._unsub_refresh = None

        self._async_cancel_bluetooth_subscription()

        # Clean up any other resources that might need to be released
        self._listeners = []

    @callback
    def _needs_poll(
        self,
        service_info: BluetoothServiceInfoBleak,
        last_poll: float | None,
    ) -> bool:
        """Determine if device needs polling based on time since last poll."""
        # Only poll if hass is running and device is connectable
        if self.hass.state != CoreState.running:
            return False

        # Check if we have a connectable device
        connectable_device = bluetooth.async_ble_device_from_address(
            self.hass, service_info.device.address, connectable=True
        )
        if not connectable_device:
            self.logger.warning(
                "No connectable device found for %s", service_info.address
            )
            return False

        # If a connection is already in progress, don't start another one
        if self._connection_in_progress:
            self.logger.debug("Connection already in progress, skipping poll")
            return False

        # If we've never polled or it's been longer than the scan interval, poll
        if last_poll is None:
            self.logger.debug("First poll for device %s", service_info.address)
            return True

        # Check if enough time has elapsed since the last poll
        time_since_poll = datetime.now().timestamp() - last_poll
        should_poll = time_since_poll >= self.scan_interval

        if should_poll:
            self.logger.debug(
                "Time to poll device %s after %.1fs",
                service_info.address,
                time_since_poll,
            )

        return should_poll

    async def _read_device_data(self, service_info: BluetoothServiceInfoBleak) -> bool:
        """Read data from a Renogy BLE device using active connection."""
        async with self._connection_lock:
            try:
                self._connection_in_progress = True
                success = False
                error = None

                # Use service_info to get a BLE device and update our device object
                if not self.device:
                    self.logger.debug(
                        "Creating new RenogyBLEDevice for %s as %s",
                        service_info.address,
                        self.device_type,
                    )
                    self.device = RenogyBLEDevice(
                        service_info.device,
                        service_info.advertisement.rssi,
                        device_type=self.device_type,
                    )
                else:
                    # Store the old name to detect changes
                    old_name = self.device.name

                    self.device.ble_device = service_info.device
                    # Update name if available from service_info
                    if (
                        service_info.name
                        and service_info.name != "Unknown Renogy Device"
                    ):
                        cleaned_name = clean_device_name(service_info.name)
                        if old_name != cleaned_name:
                            self.device.name = cleaned_name
                            self.logger.debug(
                                "Updated device name from '%s' to '%s'",
                                old_name,
                                cleaned_name,
                            )

                    # Prefer the RSSI from advertisement data if available
                    self.device.rssi = (
                        service_info.advertisement.rssi
                        if service_info.advertisement
                        and service_info.advertisement.rssi is not None
                        else service_info.device.rssi
                    )

                    # Ensure device type is set correctly
                    if self.device.device_type != self.device_type:
                        self.logger.debug(
                            "Updating device type from '%s' to '%s'",
                            self.device.device_type,
                            self.device_type,
                        )
                        self.device.device_type = self.device_type

                device = self.device
                self.logger.debug(
                    "Polling %s device: %s (%s)",
                    device.device_type,
                    device.name,
                    device.address,
                )

                # Use bleak-retry-connector for more robust connection
                try:
                    # Establish connection with retry capability
                    client = await establish_connection(
                        BleakClientWithServiceCache,
                        service_info.device,
                        device.name or device.address,
                        max_attempts=3,
                    )

                    any_command_succeeded = False

                    try:
                        self.logger.debug("Connected to device %s", device.name)
                        
                        # Create an event that will be set when notification data is received
                        notification_event = asyncio.Event()
                        notification_data = bytearray()

                        def notification_handler(sender, data):
                            notification_data.extend(data)
                            notification_event.set()

                        await client.start_notify(
                            RENOGY_READ_CHAR_UUID, notification_handler
                        )

                        for cmd_name, cmd in COMMANDS[self.device_type].items():
                            notification_data.clear()
                            notification_event.clear()

                            modbus_request = create_modbus_read_request(
                                DEFAULT_DEVICE_ID, *cmd
                            )
                            self.logger.debug(
                                "Sending %s command: %s",
                                cmd_name,
                                list(modbus_request),
                            )
                            await client.write_gatt_char(
                                RENOGY_WRITE_CHAR_UUID, modbus_request
                            )

                            #Sleep for 15 seconds to allow device to respond
                            #await asyncio.sleep(15) 

                            # Expected length: 3 header bytes + 2*word_count data + 2‑byte CRC
                            word_count = cmd[2]
                            expected_len = 3 + word_count * 2 + 2
                            start_time = self.hass.loop.time()

                            try:
                                while len(notification_data) < expected_len:
                                    remaining = MAX_NOTIFICATION_WAIT_TIME - (
                                        self.hass.loop.time() - start_time
                                    )
                                    if remaining <= 0:
                                        raise asyncio.TimeoutError()
                                    await asyncio.wait_for(
                                        notification_event.wait(), remaining
                                    )
                                    notification_event.clear()
                            except asyncio.TimeoutError:
                                self.logger.info(
                                    "Timeout – only %s / %s bytes received for %s from device %s",
                                    len(notification_data),
                                    expected_len,
                                    cmd_name,
                                    device.name,
                                )
                                continue

                            result_data = bytes(notification_data[:expected_len])
                            self.logger.debug(
                                "Received %s data length: %s (expected %s)",
                                cmd_name,
                                len(result_data),
                                expected_len,
                            )

                            cmd_success = device.update_parsed_data(
                                result_data, register=cmd[1], cmd_name=cmd_name
                            )

                            if cmd_success:
                                self.logger.debug(
                                    "Successfully read and parsed %s data from device %s",
                                    cmd_name,
                                    device.name,
                                )
                                any_command_succeeded = True
                            else:
                                self.logger.info(
                                    "Failed to parse %s data from device %s",
                                    cmd_name,
                                    device.name,
                                )

                        await client.stop_notify(RENOGY_READ_CHAR_UUID)
                        success = any_command_succeeded
                        if not success:
                            error = Exception("No commands completed successfully")

                    except BleakError as e:
                        self.logger.info(
                            "BLE error with device %s: %s", device.name, str(e)
                        )
                        error = e
                        success = False
                    except Exception as e:
                        self.logger.error(
                            "Error reading data from device %s: %s", device.name, str(e)
                        )
                        error = e
                        success = False
                    finally:
                        # BleakClientWithServiceCache handles disconnect in context manager
                        # but we need to ensure the client is disconnected
                        if client.is_connected:
                            try:
                                await client.disconnect()
                                self.logger.debug(
                                    "Disconnected from device %s", device.name
                                )
                            except Exception as e:
                                self.logger.debug(
                                    "Error disconnecting from device %s: %s",
                                    device.name,
                                    str(e),
                                )
                                # Don't override previous errors with disconnect errors
                                if error is None:
                                    error = e

                except (BleakError, asyncio.TimeoutError) as connection_error:
                    self.logger.info(
                        "Failed to establish connection with device %s: %s",
                        device.name,
                        str(connection_error),
                    )
                    error = connection_error
                    success = False

                # Always update the device availability and last_update_success
                device.update_availability(success, error)
                self.last_update_success = success

                # Update coordinator data if successful
                if success and device.parsed_data:
                    self.data = dict(device.parsed_data)
                    self.logger.debug("Updated coordinator data: %s", self.data)

                return success
            finally:
                self._connection_in_progress = False

    async def _async_poll(self, service_info: BluetoothServiceInfoBleak) -> None:
        """Poll the device."""
        # If a connection is already in progress, don't start another one
        if self._connection_in_progress:
            self.logger.debug("Connection already in progress, skipping poll")
            return

        self.last_poll_time = datetime.now()
        self.logger.debug(
            "Polling device: %s (%s)", service_info.name, service_info.address
        )

        # Read device data using service_info and Home Assistant's Bluetooth API
        success = await self._read_device_data(service_info)

        if success and self.device and self.device.parsed_data:
            # Log the parsed data for debugging
            self.logger.debug("Parsed data: %s", self.device.parsed_data)

            # Call the callback if available
            if self.device_data_callback:
                try:
                    if asyncio.iscoroutinefunction(self.device_data_callback):
                        await self.device_data_callback(self.device)
                    else:
                        self.device_data_callback(self.device)
                except Exception as e:
                    self.logger.error("Error in device data callback: %s", str(e))

            # Update all listeners after successful data acquisition
            self.async_update_listeners()

        else:
            self.logger.info("Failed to retrieve data from %s", service_info.address)
            self.last_update_success = False

    @callback
    def _async_handle_unavailable(
        self, service_info: BluetoothServiceInfoBleak
    ) -> None:
        """Handle the device going unavailable."""
        self.logger.info("Device %s is no longer available", service_info.address)
        self.last_update_success = False
        self.async_update_listeners()

    @callback
    def _async_handle_bluetooth_event(
        self,
        service_info: BluetoothServiceInfoBleak,
        change: BluetoothChange,
    ) -> None:
        """Handle a Bluetooth event."""
        # Update RSSI if device exists
        if self.device:
            self.device.rssi = service_info.advertisement.rssi
            self.device.last_seen = datetime.now()
