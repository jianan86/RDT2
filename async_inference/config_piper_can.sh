#!/usr/bin/env bash
set -euo pipefail

# Configure two Piper CAN adapters by USB bus-info.
# Left/right use operator perspective.

EXPECTED_CAN_COUNT=2
BITRATE=1000000

declare -A USB_PORTS
USB_PORTS["1-1.3:1.0"]="right_piper"
USB_PORTS["1-1.4:1.0"]="left_piper"

sudo modprobe gs_usb

current_count=$(ip -br link show type can | awk '{print $1}' | wc -l)
if [ "$current_count" -ne "$EXPECTED_CAN_COUNT" ]; then
    echo "error: detected $current_count CAN interfaces, expected $EXPECTED_CAN_COUNT" >&2
    ip -br link show type can >&2 || true
    exit 1
fi

for iface in $(ip -br link show type can | awk '{print $1}'); do
    bus_info=$(sudo ethtool -i "$iface" | awk '/bus-info/ {print $2}')
    target_name=${USB_PORTS[$bus_info]:-}
    if [ -z "$target_name" ]; then
        echo "error: unexpected CAN USB bus-info $bus_info on $iface" >&2
        exit 1
    fi

    if [ "$iface" != "$target_name" ]; then
        sudo ip link set "$iface" down
        sudo ip link set "$iface" name "$target_name"
        iface="$target_name"
    fi

    sudo ip link set "$iface" down
    sudo ip link set "$iface" type can bitrate "$BITRATE"
    sudo ip link set "$iface" up
    echo "$bus_info -> $iface bitrate=$BITRATE"
done
