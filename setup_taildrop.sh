#!/bin/sh
# Tell tailscale you are the user who should have admin rights
# to interact with tailscaled:
sudo tailscale up --operator="${USER}"

# Install a unit to systemd to run this script as you, telling
# the script where to drop the files and how to handle conflicts.
mkdir -p ~/.config/systemd/user/
TAILDROPDIR="${HOME}/Downloads"
cat <<EOF > ~/.config/systemd/user/tailreceive.service
[Unit]
Description=File Receiver Service for Taildrop

[Service]
UMask=0077
ExecStart=tailscale file get --verbose --loop "${TAILDROPDIR}"

[Install]
WantedBy=default.target

EOF

# Tell systemd to load up your new daemon and check everything is ok.
systemctl --user daemon-reload
systemctl --user start tailreceive
systemctl --user status tailreceive

