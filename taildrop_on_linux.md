# How do I set up Taildrop on Linux?

## TLDR: Run this [script](https://raw.githubusercontent.com/davideger/blog/main/setup_taildrop.sh) as you:

```sh
curl -fsSL https://raw.githubusercontent.com/davideger/blog/main/setup_taildrop.sh | sh
```

Tailscale will then automatically deposit received files into your
`Downloads` directory.  If your taildrops aren't arriving, you can
check on things at any time by running:

```sh
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
machine.  Then we tell systemd it should run tailscale file get in
loop mode to receive files to a target directory as they arrive.
Congratulations, you have now automated Taildrop receipt on your Linux box.

## Customization

If you'd like to change the directory for Taildrop downloads, or what
happens in case Tailscale receives a file that is named the same as
a file already in your Taildrop directory, change the `ExecStart` line
in `~/.config/systemd/user/tailreceive.service`.

You can for example drop files into `/home/jenny/taildrops` and overwrite
any existing same named file by changing that line to:

```
ExecStart=tailscale file get --daemon --verbose --conflict=overwrite /home/jenny/taildrops
```

For full information on available options, run `tailscale file get -h`
