---
layout:     post
title:      Linux | One Step Setup
subtitle:   A one-step setup script for newly installed Ubuntu OS
date:       2021-07-21
author:     shawn233
header-img: img/post-bg-cook.jpg
catalog:    true
tags:
    - Ubuntu
    - Linux
---

This script is a one-step setup script for newly installed Ubuntu OS. Run this script with 

```bash
chmod +x setup.sh
sudo ./setup.sh
```

Then have fun with the prepared OS for your purpose.

```bash
#!/bin/bash
USER_NAME=shawn233
HOME=/home/$USER_NAME
ALIAS_PATH=$HOME/.bash_aliases

APT_SOURCE=/etc/apt/sources.list

CLASH_VERSION=v1.6.5
CLASH_SOURCE=https://github.com/Dreamacro/clash/releases/download/${CLASH_VERSION}/clash-linux-amd64-${CLASH_VERSION}.gz
CLASH_TARGET=$HOME/Downloads/clash-installer.gz
CLASH_PATH=$HOME/Documents/clash

ANACONDA_SOURCE=https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
ANACONDA_TARGET=$HOME/Downloads/anaconda-installer.sh

CHROME_SOURCE=https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
CHROME_TARGET=$HOME/Downloads/chrome-installer.deb

testDownload() {
    # first parameter: target (a file path)
    # second parameter: source (a website)
    echo
    if test -f $1; then
        echo "$1 installer already exists"
    else 
        echo "Downloading installer from $2 ..."
        wget $2 -O $1
    fi
}

echo "Hello Ubuntu"
uname -a

# change to Tsinghua source
echo
echo "Changing to Tsinghua source ..."
if ! test -f ${APT_SOURCE}.bak; then
    mv $APT_SOURCE ${APT_SOURCE}.bak
fi
echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse" > $APT_SOURCE
echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse" >> $APT_SOURCE
echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse" >> $APT_SOURCE
echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-security main restricted universe multiverse" >> $APT_SOURCE
apt-get update

# basic
echo
echo "Installing basic pacakges ..."
apt-get install -y vim
apt-get install -y tree
apt-get install -y tmux
apt-get install -y git

echo "alias py=python" > $ALIAS_PATH
echo "alias tnew='tmux new -s wxy'" >> $ALIAS_PATH
echo "alias told='tmux attach -t wxy'" >> $ALIAS_PATH

# Clash for linux: latest release is v1.6.5 at present. Please update the version.
testDownload $CLASH_TARGET $CLASH_SOURCE
gzip -d $CLASH_TARGET
mv $HOME/Downloads/clash-installer $CLASH_PATH
chmod +x $CLASH_PATH
echo "alias clash='${CLASH_PATH}'" >> $ALIAS_PATH
echo "alias clash-fetch='wget https://cylink.moe/link/pxkV13bzyDmshqcX?clash=1 -O ${HOME}/.config/clash/config.yaml'" >> $ALIAS_PATH

# Anaconda
testDownload $ANACONDA_TARGET $ANACONDA_SOURCE
chmod +x $ANACONDA_TARGET
$ANACONDA_TARGET

# Chrome
testDownload $CHROME_TARGET $CHROME_SOURCE
apt install $CHROME_TARGET

# VSCode

```