---
layout:     post
title:      Linux | File Compression and Extraction
subtitle:   using zip, tar, rar, and 7z
date:       2020-04-20
author:     Xinyu Wang
header-img: img/post-bg-cook.jpg
catalog:    true
tags:
    - Linux
---

# Introduction

We compress files into an archive, or extract from one, in the following scenarios:

- On PC, we free the storage by putting a set of files into an archive.
- On PC, we download from the Internet and get compressed files.
- On servers, we make the best of the network bandwidth, by reducing the transmission payload.
- On business cloud storage, we avoid paying for more storage by compressing files into an archive.
- Whenever we want to access files in an archive, an uncompression is indispensible.

This article focuses on the Linux platform, which is widely deployed on PC and servers as the operating system (OS). PC OS includes Ubuntu, CentOS, Redhat, etc. On servers, Linux as OS is almost prevailing, such as Colab, Amazon Cloud and Ali Cloud.

By reading this article, you will learn the basic usage of four common compression tools, namely, zip, tar, rar, and 7z.

# Installation

(skip if you've already installed)

 

# ZIP



# TAR

# RAR

# 7Z

# Comparison

```
# compare zip, 7z and tar
# !7z a -t7z -r download-actor.7z ./download
# !tar c -czf tmp.tar.gz ./download

# !ls -hl
# total 2.2G
# drwxr-xr-x 3 root root 4.0K Mar 24 10:21 download
# -rw-r--r-- 1 root root 675M Apr 20 03:04 download-actor.7z
# -rw-r--r-- 1 root root 679M Apr 20 03:07 download-actor.tar.gz
# -rw------- 1 root root 690M Apr 20 02:57 download-actor.zip
# -rw-r--r-- 1 root root 3.7K Apr 20 02:51 download-official.py
# -rw-r--r-- 1 root root  11M Apr 20 02:51 facescrub_actors_shuffled.txt
# drwxr-xr-x 1 root root 4.0K Apr  3 16:24 sample_data
# -rw-r--r-- 1 root root 154M Apr 20 03:27 tmp.tar.gz

# from google.colab import drive
# drive.mount("/content/drive")
# !cp download-actor.zip /content/drive/My\ Drive/sync/NNInversion/
# !rm /content/drive/My\ Drive/sync/NNInversion/download-actor.7z
# drive.flush_and_unmount()
```



# Conclusion



