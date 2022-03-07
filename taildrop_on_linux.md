# How do I set up Taildrop on Linux?

## TLDR: Copy and paste these commands:

```sh
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
```

## Background and explanation

By default, Tailscale on Linux runs as `root` and receives any taildrop
files to a staging directory it controls.  To collect waiting files
you can run the following command:

```sh
sudo tailscale file get "/home/$USER/Downloads"
```

However, this is problematic for a few reasons:

1. **Unnecessary root usage** [#2324](https://github.com/tailscale/tailscale/issues/2324)

   You probably think of your Linux workstation as a single user box, so
   why should you have to `sudo` just to get your files?  You don't
   have to `sudo` whenever you download a file using your web browser,
   and neither do you have to become "admin" on Windows to receive your
   Taildrops.  Running this command as `sudo` means that you're enabling
   it to possibly overwrite any file on your system (sudo has permissions,
   after all).

   Imagine someone figures out how to run a small script as you that
   symlinks `/home/$USER/Downloads/essay.pdf` to `/etc/passwd` and later
   you taildrop `essay.pdf` to your linux box.  If you were to run the
   above command, would you be accidentally overwriting `/etc/passwd`?
   Answer: No, but this is the sort of attack that should make you think
   twice about running things as root willy nilly.  If the attacker
   was able to replace your `Downloads` file with a symlink to `/etc`
   however, and you ran `sudo tailscale file get ${HOME}/Downloads/`
   you *would* be liable to overwrite files in `/etc`!


2. **No automatic downloads** [#2312](https://github.com/tailscale/tailscale/issues/2312)

   It's sort of annoying to not have files automatically delivered to
   a standard directory you can access.

3. **Lost file timestamps** [#2057](https://github.com/tailscale/tailscale/issues/2057)

   You're probably going to share a bunch of individual files and then
   sometime later remember to "receive" them all.  At that point, you'll
   lose record of when tailscaled actually received them, and they'll all
   get the timestamp of when you ran `tailscale file get`

To overcome these problems, first we inform `tailscaled` which
user should have unfettered access to tailscale and taildrops on this
machine.  Then we set up a script to sit in a loop and receive files
to a target directory as they arrive.  Congratulations, you have now
automated Taildrop receipt on your Linux box.
