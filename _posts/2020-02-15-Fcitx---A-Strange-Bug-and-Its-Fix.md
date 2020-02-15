---
layout:     post
title:      Fcitx | A Strange Bug and Its Fix
subtitle:   fcitx crash receiving signal no. 11
date:       2020-02-15
author:     Xinyu Wang
header-img: img/post-bg-cook.jpg
catalog:    true
tags:
    - Ubuntu
---

There was a bug which had troubled me for quite a few days. Every time I used Google pinyu to type in Chinese characters, fcitx crashed and the only characters that I could type were English. 

When I checked the crash log of fcitx,

```bash
vim ~/.config/fcitx/log/crash.log
```

it says:

```bash
=========================
FCITX 4.2.9.6 -- Get Signal No.: 11
Date: try "date -d @1581747265" if you are using GNU date ***
ProcessID: 15354
fcitx(+0x1627)[0x562e41e83627]
/lib/x86_64-linux-gnu/libc.so.6(+0x3ef20)[0x7fb8e072bf20]
/usr/lib/x86_64-linux-gnu/libgooglepinyin.so.0(_ZN10ime_pinyin8UserDict23locate_first_in_offsetsEPKNS0_18UserDictSearchableE+0x5f)[0x7fb8d28b93cf]
/usr/lib/x86_64-linux-gnu/libgooglepinyin.so.0(_ZN10ime_pinyin8UserDict9_get_lpisEPKttPNS_10LmaPsbItemEmPb+0x225)[0x7fb8d28bb4e5]
/usr/lib/x86_64-linux-gnu/libgooglepinyin.so.0(_ZN10ime_pinyin8UserDict11extend_dictEtPKNS_11DictExtParaEPNS_10LmaPsbItemEmPm+0x3e)[0x7fb8d28bb74e]
/usr/lib/x86_64-linux-gnu/libgooglepinyin.so.0(_ZN10ime_pinyin12MatrixSearch10extend_dmiEPNS_11DictExtParaEPNS_13DictMatchInfoE+0x23f)[0x7fb8d28b1f2f]
/usr/lib/x86_64-linux-gnu/libgooglepinyin.so.0(_ZN10ime_pinyin12MatrixSearch15add_char_qwertyEv+0x359)[0x7fb8d28b2699]
/usr/lib/x86_64-linux-gnu/libgooglepinyin.so.0(_ZN10ime_pinyin12MatrixSearch6searchEPKcm+0xd9)[0x7fb8d28b3fb9]
/usr/lib/x86_64-linux-gnu/libgooglepinyin.so.0(im_search+0x1e)[0x7fb8d28b5dae]
/usr/lib/x86_64-linux-gnu/fcitx/fcitx-googlepinyin.so(+0x2610)[0x7fb8d2aef610]
/usr/lib/x86_64-linux-gnu/fcitx/fcitx-googlepinyin.so(+0x27ee)[0x7fb8d2aef7ee]
/usr/lib/x86_64-linux-gnu/libfcitx-core.so.0(FcitxInstanceProcessKey+0x5f1)[0x7fb8e113e841]
/usr/lib/x86_64-linux-gnu/fcitx/fcitx-ipc.so(+0x5f97)[0x7fb8d1b7ef97]
/lib/x86_64-linux-gnu/libdbus-1.so.3(+0x22d90)[0x7fb8df4cfd90]
/lib/x86_64-linux-gnu/libdbus-1.so.3(dbus_connection_dispatch+0x32a)[0x7fb8df4c0b5a]
/usr/lib/x86_64-linux-gnu/fcitx/fcitx-dbus.so(+0x2408)[0x7fb8df6fc408]
/usr/lib/x86_64-linux-gnu/fcitx/fcitx-dbus.so(+0x2523)[0x7fb8df6fc523]
/usr/lib/x86_64-linux-gnu/libfcitx-core.so.0(+0x8acc)[0x7fb8e1131acc]
/usr/lib/x86_64-linux-gnu/libfcitx-core.so.0(FcitxInstanceRun+0x330)[0x7fb8e1132560]
fcitx(+0xf6c)[0x562e41e82f6c]
/lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0xe7)[0x7fb8e070eb97]
fcitx(+0xfea)[0x562e41e82fea]
```

I searched this crash log on Google, but found none similar previous cases. Some suggested a re-installation, and others mentioned some strange term, Wayland. I thought this as a small problem and not worth so much trouble. So I tried a simple tip in a thread which solved another crash problem of fcitx. Fortunately, it worked.

So now I post this thread, hoping to help those who also have this problem, and save them time to come up with the solution. Now, my fix is to remove the configuration folder of the abnormal input method, google pinyu, by using the following command.

```bash
rm -r ~/.config/fcitx/googlepinyin
```

Hope this thread helps~
