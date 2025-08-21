"""Switch entity for Renogy BLE load control."""

import asyncio

from homeassistant.components.switch import SwitchEntity
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import EntityCategory

from .ble import RenogyActiveBluetoothCoordinator, RenogyBLEDevice
from .const import LOGGER, DOMAIN

async def async_setup_entry(hass: HomeAssistant, config_entry, async_add_entities):
    """Set up the Renogy load switch entity from a config entry."""
    LOGGER.debug("Setting up Renogy load switch entity")
    coordinator: RenogyActiveBluetoothCoordinator = hass.data["renogy"][config_entry.entry_id]["coordinator"]
    entities = [RenogyLoadSwitch(coordinator)]
    async_add_entities(entities)

class RenogyLoadSwitch(SwitchEntity):
    """Representation of the Renogy load switch."""

    _attr_entity_category = EntityCategory.CONFIG

    def __init__(self, coordinator: RenogyActiveBluetoothCoordinator):
        self.coordinator = coordinator
        device: RenogyBLEDevice = coordinator.device
        device_name = device.name if device and device.name else "Renogy"
        device_address = getattr(device, "address", "unknown")
        self._attr_name = f"{device_name} Load Switch"
        self._attr_unique_id = f"{device_address}_load_status"
        # Don't set _attr_is_on here since we override the is_on property
        self._attr_device_info = {
            "identifiers": {(DOMAIN, device_address)},
            "name": device_name,
            "manufacturer": "Renogy",
            "model": getattr(device, "model", "Unknown"),
        }

    async def async_added_to_hass(self):
        """Subscribe to coordinator updates when added to hass."""
        await super().async_added_to_hass()
        self.async_on_remove(
            self.coordinator.async_add_listener(self.async_write_ha_state)
        )

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return self.coordinator.last_update_success

    @property
    def is_on(self) -> bool:
        """Return True if load is ON."""
        device = self.coordinator.device
        if device and device.parsed_data:
            load_status = device.parsed_data.get("load_status")
            return load_status == "on"
        return False

    async def async_turn_on(self, **kwargs):
        """Turn the load ON."""
        LOGGER.info("User requested to turn load ON for device %s", self.coordinator.device.name)
        device = self.coordinator.device
        if not device:
            LOGGER.error("No device available for load control")
            return
            
        # Store the current state before attempting to change it
        old_state = self.is_on
        LOGGER.debug("Load switch current state before command: %s", "ON" if old_state else "OFF")
        
        success = await device.async_set_load(True)
        LOGGER.info("Load turn ON command result: %s", success)
        
        if success:
            # Give device time to process the command before refreshing
            await asyncio.sleep(2.0)  # Increased wait time
            await self.coordinator.async_request_refresh()
            
            # Wait a bit for the refresh to complete
            await asyncio.sleep(1.0)
            
            # Check if the state actually changed
            new_state = self.is_on
            LOGGER.debug("Load switch state after command and refresh: %s", "ON" if new_state else "OFF")
            
            if new_state and not old_state:
                LOGGER.info("✓ Load successfully turned ON - state change confirmed")
            elif new_state and old_state:
                LOGGER.info("✓ Load was already ON - no state change needed")
            else:
                LOGGER.warning("✗ Load may not have turned ON - state did not change as expected")
                LOGGER.warning("Device may not support load control or may require manual load connection")
                
            self.async_write_ha_state()
        else:
            LOGGER.error("Failed to turn load ON for device %s", device.name)

    async def async_turn_off(self, **kwargs):
        """Turn the load OFF."""
        device = self.coordinator.device
        if not device:
            LOGGER.error("No device available for load control")
            return
            
        LOGGER.info("User requested to turn load OFF for device %s", device.name)
        
        # Store the current state before attempting to change it
        old_state = self.is_on
        LOGGER.debug("Load switch current state before command: %s", "ON" if old_state else "OFF")
        
        success = await device.async_set_load(False)
        LOGGER.info("Load turn OFF command result: %s", success)
        
        if success:
            # Give device time to process the command before refreshing
            await asyncio.sleep(2.0)  # Increased wait time
            await self.coordinator.async_request_refresh()
            
            # Wait a bit for the refresh to complete
            await asyncio.sleep(1.0)
            
            # Check if the state actually changed
            new_state = self.is_on
            LOGGER.debug("Load switch state after command and refresh: %s", "ON" if new_state else "OFF")
            
            if not new_state and old_state:
                LOGGER.info("✓ Load successfully turned OFF - state change confirmed")
            elif not new_state and not old_state:
                LOGGER.info("✓ Load was already OFF - no state change needed")
            else:
                LOGGER.warning("✗ Load may not have turned OFF - state did not change as expected")
                LOGGER.warning("Device may not support load control or may require manual load disconnection")
                
            self.async_write_ha_state()
        else:
            LOGGER.error("Failed to turn load OFF for device %s", device.name)