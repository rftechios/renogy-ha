"""Switch entity for Renogy BLE load control."""

from homeassistant.components.switch import SwitchEntity
from homeassistant.core import HomeAssistant
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
        self._attr_is_on = False
        self._attr_device_info = {
            "identifiers": {(DOMAIN, device_address)},
            "name": device_name,
            "manufacturer": "Renogy",
            "model": getattr(device, "model", "Unknown"),
        }

    @property
    def is_on(self) -> bool:
        """Return True if load is ON."""
        device = self.coordinator.device
        if device and device.parsed_data:
            return bool(device.parsed_data.get("load_status", False))
        return self._attr_is_on

    async def async_turn_on(self, **kwargs):
        """Turn the load ON."""
        device = self.coordinator.device
        if device:
            success = await device.async_set_load(True)
            if success:
                self._attr_is_on = True
                await self.coordinator.async_request_refresh()
                self.async_write_ha_state()

    async def async_turn_off(self, **kwargs):
        """Turn the load OFF."""
        device = self.coordinator.device
        if device:
            success = await device.async_set_load(False)
            if success:
                self._attr_is_on = False
                await self.coordinator.async_request_refresh()
                self.async_write_ha_state()

    async def async_update(self):
        """Update the switch state."""
        await self.coordinator.async_request_refresh()
        self._attr_is_on = self.is_on