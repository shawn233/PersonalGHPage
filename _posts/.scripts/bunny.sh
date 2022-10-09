#!/bin/bash
# Ubuntu Bunny Initialization
# created by shawn233
# Feb 2022
# last updated: Jul 2022

# Ubuntu bash scripting beginner's guide: https://help.ubuntu.com/community/Beginners/BashScripting

# How to use this script:
# (0) When installing Ubuntu, you can skip all the downloading tasks. Those packages
#     will be installed in this script, from a better source.
#     If you are using VMWare Workstation, remember to install 'open-vm-tools' first. 
#     This will allow you to copy the script into the virtual machine.
# (1) check all the TODO items in the script
# (2) go to the working directory where the script is
# (3) grant the execution permission: `chmod +x bunny.sh`
# (4) run the script: `./bunny.sh`

# Note:
# - In bash scripting value assignment must be: `name=value` (no blankspace), 
#   otherwise bash will think it as separate commands. For instance, 
#   `name= value` will be taken as `name=` and `value`.

# 0. define variables
PROMPT=">> "
USER_NAME=$USER
HOME_DIR=$HOME
PWD=$(pwd)
PKG_DIR=$PWD/pkgs
mkdir -p ${PKG_DIR}
echo "Hello, $USER_NAME ($HOME_DIR). Welcome to bunny initialization"

APT_PATH=/etc/apt/sources.list
APT_BACKUP_PATH=$APT_PATH.init.bak
APT_SOURCE=ustc # [tuna, tsinghua, ali, aliyun, ustc, default]
APT_CONF_DIR=/etc/apt/apt.conf.d

# pkgs to be installed from apt
# install third-party software: https://askubuntu.com/questions/290293/how-can-i-install-the-third-party-software-option-after-ive-skipped-it-in-the
PKG_LIST="build-essential gcc g++ cmake vim git tree tmux curl net-tools htop"
# UPDATE: now ubuntu-restricted-extras will be handled after CLASH (for network reasons)
# 2022/07/01: ubuntu-restricted-extras causes apt-get to hang because ttf-mscorefonts-installer acquires extra data from sourceforge.net, which in some cases without proxy are very time-consuming.
# This issue is fixed by adding a proxy setup (can be named as 01proxy) to "/etc/apt/apt.conf.d/".
# This issue is related to a system script /usr/lib/update-notifier/package-data-downloader.
# This script invokes apt-helper to download extra package data, which does not use the system proxy.

# set NO_<SOFTWARE> 1 to cancel a software installation
NO_CLASH=0 # set 'NO_CLASH' to 1 may cause the rest to fail due to network issues
IS_GITHUB_ACCESSIBLE=1 # TODO: set to 1 if github is accessible
# check https://github.com/Fndroid/clash_for_windows_pkg/releases for latest
CLASH_DOWNLOAD=https://github.com/Fndroid/clash_for_windows_pkg/releases/download/0.19.23/Clash.for.Windows-0.19.23-x64-linux.tar.gz # may be inaccessible without proxy, so I use jbox
CLASH_JBOX_DOWNLOAD=https://jbox.sjtu.edu.cn:10081/v2%2Fdelivery%2Fdata%2F5e6c89a00dc54ef29cda2e13651e4c05%2F? # clash package shared from my jbox
CLASH_PROFILE_LINK= # TODO: fill in a clash profile link

NO_TOMATOSHELL=0

NO_VSCODE=0
VSCODE_DOWNLOAD=https://code.visualstudio.com/sha/download?build=stable\&os=linux-deb-x64

NO_CHROME=0
CHROME_DOWNLOAD=https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb

NO_ANACONDA=0
# TODO: check https://www.anaconda.com/products/individual or https://repo.anaconda.com/archive/ for latest
ANACONDA_DOWNLOAD=https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh

# if this script is running as root, all the new files and directories will belong to root.
# TODO: check the Ubuntu version
echo -e "[Ubuntu 20.04 STL] Make sure you have a reliable network connection and not running in \"sudo\""
echo
echo "Please check the following variables are correctly assigned:"
echo "- CLASH_DOWNLOAD (if accessible, set IS_GITHUB_ACCESSIBLE to 1)"
echo "- IS_GITHUB_ACCESSIBLE=$IS_GITHUB_ACCESSIBLE"
echo "- CLASH_JBOX_DOWNLOAD (make sure it is accessible if github is not)"
echo "- CLASH_PROFILE_LINK (fill in a clash profile link)"
echo "- ANACONDA_DOWNLOAD (check https://www.anaconda.com/products/individual for latest)"
echo
echo "Press any key to start, or [Ctrl+C] to abort"
read -n 1 -p "$PROMPT"


# 1. change apt source
echo
echo "backing up the default apt source ..."
echo -e "- apt source path:\t$APT_PATH\n- backup path:\t\t$APT_BACKUP_PATH"
if ! [ -e $APT_BACKUP_PATH ]
then
	sudo cp $APT_PATH $APT_BACKUP_PATH # backup the default source
	echo "done"
else
	echo "file existed, skip"
fi

echo
echo "writing new apt source ..."
echo -e "- source selection:\t$APT_SOURCE"

# TODO: update apt sources to date (a different Ubuntu version uses an individual source)
# - tuna source: https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/
TUNA_SOURCE="
# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-security main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-security main restricted universe multiverse

# 预发布软件源，不建议启用
# deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-proposed main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-proposed main restricted universe multiverse
"
# - aliyun source: https://developer.aliyun.com/mirror/ubuntu
ALIYUN_SOURCE="
deb http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse

deb http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse

deb http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse

deb http://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse

deb http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse
"
# - USTC source: https://mirrors.ustc.edu.cn/help/ubuntu.html
USTC_SOURCE="
# 默认注释了源码仓库，如有需要可自行取消注释
deb https://mirrors.ustc.edu.cn/ubuntu/ focal main restricted universe multiverse
# deb-src https://mirrors.ustc.edu.cn/ubuntu/ focal main restricted universe multiverse

deb https://mirrors.ustc.edu.cn/ubuntu/ focal-security main restricted universe multiverse
# deb-src https://mirrors.ustc.edu.cn/ubuntu/ focal-security main restricted universe multiverse

deb https://mirrors.ustc.edu.cn/ubuntu/ focal-updates main restricted universe multiverse
# deb-src https://mirrors.ustc.edu.cn/ubuntu/ focal-updates main restricted universe multiverse

deb https://mirrors.ustc.edu.cn/ubuntu/ focal-backports main restricted universe multiverse
# deb-src https://mirrors.ustc.edu.cn/ubuntu/ focal-backports main restricted universe multiverse

# 预发布软件源，不建议启用
# deb https://mirrors.ustc.edu.cn/ubuntu/ focal-proposed main restricted universe multiverse
# deb-src https://mirrors.ustc.edu.cn/ubuntu/ focal-proposed main restricted universe multiverse
"

apt_target=${PKG_DIR}/${APT_SOURCE}.apt_target # my strange way to avoid repetition 
# Note: sudo echo "$TUNA_SOURCE" > $APT_PATH results in Permission Denied because the > operator is not privileged
if ! [ -e $apt_target ]
then
	rm -f ${PKG_DIR}/*.apt_target
	case $APT_SOURCE in 
	tsinghua | tuna)
		touch ${PKG_DIR}/tuna.apt_target ${PKG_DIR}/tsinghua.apt_target
		echo "$TUNA_SOURCE" > $apt_target
		sudo cp $apt_target $APT_PATH
		echo "done (tsinghua)"
		;;
	ali | aliyun)
		touch ${PKG_DIR}/ali.apt_target ${PKG_DIR}/aliyun.apt_target
		echo "$ALIYUN_SOURCE" > $apt_target
		sudo cp $apt_target $APT_PATH
		echo "done (aliyun)"
		;;
	ustc)
		touch ${PKG_DIR}/ustc.apt_target
		echo "$USTC_SOURCE" > $apt_target
		sudo cp $apt_target $APT_PATH
		echo "done (ustc)"
		;;
	default)
		touch ${PKG_DIR}/default.apt_target
		sudo cp $APT_BACKUP_PATH $APT_PATH	
		echo "done (default)" 
		;;
	*)
		touch ${PKG_DIR}/default.apt_target
		sudo cp $APT_BACKUP_PATH $APT_PATH
		echo "done (unknown, default taken)" 
		;;
	esac
	# only update when apt source is changed
	sudo apt-get update
	sudo apt-get -y upgrade
else
	echo "already as selected, skip"
fi



# 2. install packages
echo
echo "installing new packages ..."
echo "- packages to install: $PKG_LIST"
sudo apt-get -y install $PKG_LIST
echo "done"


# 3. set bash aliases
echo
echo "setting bash aliases ..."
bash_aliases_target=${HOME_DIR}/.bash_aliases
MY_BASH_ALIASES="alias py='python'
alias cfw='${PWD}/cfw/cfw'
alias tnew='tmux new -s wxy'
alias told='tmux at -t wxy'
alias tomato='tomatoshell -t 40 -d 20 -f'
"
if ! [ -e $bash_aliases_target ]
then
	echo "- my bash aliases"
	echo "$MY_BASH_ALIASES"
	echo "writing to $bash_aliases_target"
	echo "$MY_BASH_ALIASES" > $bash_aliases_target
	source ${HOME_DIR}/.bashrc
	echo "done"
else 
	echo "$bash_aliases_target existed, skip"
fi


# 4. install common softwares
echo
echo "installing common softwares ..."
echo "if any network issue occurs, try to re-run this script, or download manually"
echo "NOTE: if curl is aborted during download, please remove the downloaded file"
mkdir -p $PKG_DIR

# clash (clash_for_windows_pkg)
# github: https://github.com/Fndroid/clash_for_windows_pkg/releases
echo
echo "- clash"
clash_target=${PWD}/cfw
clash_pkg=${PKG_DIR}/clash.tar.gz
if ! [ -d $clash_target ]
then
	if [ $NO_CLASH -eq 0 ]
	then
		# (a) download clash from an accessible source (w/o proxy)
		# here I use jbox
		if ! [ -e $clash_pkg ]
		then
			if ! [ $IS_GITHUB_ACCESSIBLE -eq 1 ]
			then
				# download from jbox
				echo "downloading from ${CLASH_JBOX_DOWNLOAD}"
				curl -o $clash_pkg -L ${CLASH_JBOX_DOWNLOAD}
			else
				# directly download from GITHUB if possible
				echo "downloading from ${CLASH_DOWNLOAD}"
				curl -o $clash_pkg -L ${CLASH_DOWNLOAD}
			fi
		else
			echo "clash package existed, skip download"
		fi
		# (b) extract from clash.tar.gz
		echo "extracting package at $clash_pkg"
		tar -xzf $clash_pkg
		mv Clash\ for\ Windows* $clash_target
		# (c) remove clash package
		if [ -d $clash_target ]
		then
			# rm -f $clash_pkg
			echo "done"
		fi
	elif [ $NO_CLASH -eq 1 ]
	then
		echo "[warning] skip installing clash may cause network issues!"
		echo "[warning] to install clash, set var 'NO_CLASH' to 1"
		echo "Press any key to continue w/o clash, [Ctrl+C] to abort"
		read -n 1 -p "$PROMPT"
	else
		echo "[error] invalid value: NO_CLASH=$NO_CLASH, only 0 or 1 is allowed"
		echo "abort"
		exit
	fi
else
	echo "clash existed, skip"
fi

# launch clash
echo -n "launching clash ... "
# reference: https://askubuntu.com/questions/157779/how-to-determine-whether-a-process-is-running-or-not-and-make-use-it-to-make-a-c
if ! pgrep -x "cfw" > /dev/null
then
	$clash_target/cfw &
	echo "done"
else
	echo "already running, skip"
fi

echo -n "configuring proxy settings ... "
if [ "$HTTP_PROXY" = "http://127.0.0.1:7890/" ] && [ "$HTTPS_PROXY" = "http://127.0.0.1:7890/" ]
then
	echo "already configured, skip"
else
	echo
	# configure clash
	echo "Please configure clash manually. Here are the instructions:"
	echo -e "(1) [General] turn on \"Start with Linux\" if needed"
	echo -e "(2) [Profiles] input $CLASH_PROFILE_LINK and click \"Download\""
	echo -e "(3) [Profiles] switch to the new profile if needed"
	echo -e "(4) close the window"
	echo "Press any key to continue"
	read -n 1 -p "$PROMPT"

	# set system proxy
	echo "Please configure system proxy manually. Here are the instructions:"
	echo "(1) click open Settings > Network > Network Proxy"
	echo "(2) swith from Disabled to Manual"
	echo "(3) set HTTP Proxy, HTTPS Proxy and Socks Host to 127.0.0.1 with port 7890"
	echo "(4) close the windows"
	echo "Press any key to continue"
	read -n 1 -p "$PROMPT"

	# set terminal proxy
	export HTTP_PROXY=http://127.0.0.1:7890/
	export HTTPS_PROXY=http://127.0.0.1:7890/
	echo "set bash proxy to HTTP_PROXY=$HTTP_PROXY HTTPS_PROXY=$HTTPS_PROXY"
	echo "done"
	
	# set apt proxy
	echo "set apt proxy to http://127.0.0.1:7890/"
	touch ${PKG_DIR}/01proxy
	echo -e "Acquire::http::Proxy \"http://127.0.0.1:7890/\";" > ${PKG_DIR}/01proxy
	sudo cp ${PKG_DIR}/01proxy ${APT_CONF_DIR}/
	echo "done"
fi

# ubuntu-restricted-extras 
# (see PKG_LIST comments for info)
echo
echo "- ubuntu-restricted-extras"
sudo apt-get -y install ubuntu-restricted-extras
echo "done"

# tomatoshell
# credit: https://github.com/LytixDev
# source: https://github.com/LytixDev/tomatoshell
echo
echo "- LityxDev/tomatoshell"
tomatoshell_target=/usr/local/bin/tomatoshell
tomatoshell_pkg=${PWD}/tomatoshell # a folder
if ! [ -e chrome_target ]
then
	if [ $NO_TOMATOSHELL -eq 0 ]
	then
		# (a) install preliminary packages
		sudo apt-get install -y libasound2 gawk figlet pulseaudio mpv
		# (b) download repo from Github
		if ! [ -e $tomatoshell_pkg ]
		then
			echo "downloading from Github"
			git clone git@github.com:LytixDev/tomatoshell.git
		else
			echo "tomatoshell package existed, skip download"
		fi
		# (c) install tomatoshell
		cd tomatoshell && ./configure install
		cd ${PWD}
		# (d) verify installation
		if [ -e $tomatoshell_target ]
		then
			echo "done"
		else
			echo "[error] weird... tomatoshell installation failed, please retry"
			echo "abort"
			exit
		fi
	elif [ $NO_TOMATOSHELL -eq 1 ]
	then
		echo "tomatoshell installation is skipped since NO_TOMATOSHELL is set"
	else
		echo "[error] invalid value: NO_TOMATOSHELL=$NO_TOMATOSHELL, only 0 or 1 is allowed"
		echo "abort"
		exit
	fi 
else
	echo "tomatoshell existed, skip"
fi

# vscode
echo
echo "- vscode"
vscode_target=/bin/code
vscode_pkg=${PKG_DIR}/vscode.deb
if ! [ -e $vscode_target ]
then
	if [ $NO_VSCODE -eq 0 ]
	then
		# (a) download vscode from official site
		if ! [ -e $vscode_pkg ]
		then
			echo "downloading from $VSCODE_DOWNLOAD"
			curl -o $vscode_pkg -L $VSCODE_DOWNLOAD
		else
			echo "vscode package existed, skip download"
		fi
		# (b) install vscode
		echo "installing"
		sudo apt install $vscode_pkg
		# (c) remove vscode package
		if [ -e $vscode_target ]
		then
			# rm -f $vscode_pkg
			echo "done"
		fi
	elif [ $NO_VSCODE -eq 1 ]
	then
		echo "vscode installation is skipped since NO_VSCODE is set"
	else
		echo "[error] invalid value: NO_VSCODE=$NO_VSCODE, only 0 or 1 is allowed"
		echo "abort"
		exit
	fi 
else
	echo "vscode existed, skip"
fi

# chrome
echo
echo "- chrome"
chrome_target=/bin/google-chrome
chrome_pkg=${PKG_DIR}/chrome.deb
if ! [ -e $chrome_target ]
then
	if [ $NO_CHROME -eq 0 ]
	then
		# (a) download chrome from official site
		if ! [ -e $chrome_pkg ]
		then
			echo "downloading from $CHROME_DOWNLOAD"
			curl -o $chrome_pkg -L $CHROME_DOWNLOAD
		else
			echo "chrome package existed, skip download"
		fi
		# (b) install chrome
		echo "installing"
		sudo apt install $chrome_pkg
		# (c) remove chrome package
		if [ -e $chrome_target ]
		then
			# rm -f $chrome_pkg
			echo "done"
		fi
	elif [ $NO_CHROME -eq 1 ]
	then
		echo "chrome installation is skipped since NO_CHROME is set"
	else
		echo "[error] invalid value: NO_CHROME=$NO_CHROME, only 0 or 1 is allowed"
		echo "abort"
		exit
	fi 
else
	echo "chrome existed, skip"
fi


# anaconda
echo
echo "- anaconda"

# TODO: check https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/ for latest
ANACONDA_TUNA_SOURCE="
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
"

anaconda_target=${HOME_DIR}/anaconda3
anaconda_pkg=${PKG_DIR}/anaconda.sh
if ! [ -d $anaconda_target ]
then
	if [ $NO_ANACONDA -eq 0 ]
	then
		# (a) download anaconda from official site
		if ! [ -e $anaconda_pkg ]
		then
			echo "downloading from $ANACONDA_DOWNLOAD"
			curl -o $anaconda_pkg -L $ANACONDA_DOWNLOAD
		else
			echo "anaconda package existed, skip download"
		fi
		# (b) install anaconda
		# reference: https://docs.anaconda.com/anaconda/install/linux/
		echo "installing"
		sudo apt-get -y install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 # as suggested in the official guide
		bash $anaconda_pkg
		source ${HOME_DIR}/.bashrc
		# (c) remove anaconda package
		if [ -e $anaconda_target ]
		then
			# rm -f $anaconda_pkg
			echo "done"
		fi
		# (d) change Anaconda source to tuna
		echo "$ANACONDA_TUNA_SOURCE" > ~/.condarc
		echo "Anaconda source changed to tuna"
		# (e) anaconda setup
		# conda cheat sheet: https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html
		py_env_name=py
		py_pkg_list="numpy pandas scipy matplotlib joblib requests pyqt"
		echo "creating a new environment $py_env_name with startup packages: $py_pkg_list"
		# Seems that even with source .bashrc, the terminal cannot find the command `conda`. May need to change PATH
		source ~/.bashrc
		conda create --name $py_env_name python $py_pkg_list
		echo "done, instructions to auto activate $py_env_name:"
		echo -e "(1) run \"conda config --set auto_activate_base false\""
		echo -e "(2) add \"conda activate $py_env_name\" to ${HOME_DIR}/.bashrc, after the conda initialization block"
	elif [ $NO_ANACONDA -eq 1 ]
	then
		echo "anaconda installation is skipped since NO_ANACONDA is set"
	else
		echo "[error] invalid value: NO_ANACONDA=$NO_ANACONDA, only 0 or 1 is allowed"
		echo "abort"
		exit
	fi 
else
	echo "anaconda existed, skip"
fi

echo
echo "Thank you for using bunny initialization. Author: shawn233" 

