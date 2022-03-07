#!/bin/sh
# Tell tailscale you are the user who should have admin rights
# to interact with tailscaled:
sudo tailscale up --operator="${USER}"

# Install a script to loop, receiving files to a given directory.
mkdir -p ~/bin/
cat <<EOF > ~/bin/tailreceive.sh
#!/bin/sh

# Continuously receive Taildrop files to a given directory.
while true
do
	if ! tailscale file get --wait "\$@"; then
		# Avoid spin-looping trying to receive files we just failed on.
		# It would be great if there was a signal that there's actually
		# a new file somehow and not just the ones we already tried.
		echo "Error receiving taildrops.  Pausing for 10 seconds."
		sleep 10
	else
		echo "Taildrop received files successfully."
	fi
done
EOF
chmod u+rx ~/bin/tailreceive.sh

# Install a unit to systemd to run this script as you, telling
# the script where to drop the files and how to handle conflicts.
mkdir -p ~/.config/systemd/user/
TAILDROPDIR="${HOME}/Downloads"
cat <<EOF > ~/.config/systemd/user/tailreceive.service
[Unit]
Description=File Receiver Service for Taildrop

[Service]
UMask=0077
ExecStart=${HOME}/bin/tailreceive.sh --verbose --conflict=rename "${TAILDROPDIR}"

[Install]
WantedBy=default.target

EOF

# Tell systemd to load up your new daemon and check everything is ok.
systemctl --user daemon-reload
systemctl --user start tailreceive
systemctl --user status tailreceive

