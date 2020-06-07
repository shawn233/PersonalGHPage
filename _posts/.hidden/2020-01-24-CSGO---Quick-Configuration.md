---
layout:     post
title:      CSGO | Quick Configuration
subtitle:   My configuration script
date:       2020-01-24
author:     Xinyu Wang
header-img: img/post-bg-cook.jpg
catalog:    true
tags:
    - CS:GO
---

This post shows my configuration codes to quickly configure my CS:GO settings. Some settings are commonly useful and recommended for every one, while others are personal and should be used with consideration.

## Launch Options

```bash
-console -novid -exec autoexec -high -fullscreen -width 1280 -height 720 -perfectworld
```

Enable console, disable intro video, give the game high CPU priority, and load in full screen mode


## Generic Settings

```bash
// Display
mat_monitorgamma "1.6"          // lightness
sys_antialiasing "0"            // anti aliasing
fps_max 300                     // max fps
//r_dynamic "0"                   // disable dynamic lighting
r_drawparticles 1             	// disable particles
r_drawtracers_firstperson 1   	// disable trace fires
muzzleflash_light 1             // diable the flash of shooting
r_eyemove 0                     // makes eye motionless
r_gloss 0                       // turn off eye shine
cl_downloadfilter nosounds      // block downloads of custom sounds from server
mat_savechanges                 // save changes
con_enable 1                    // enable console
cl_autohelp "1"                 // activates help messages

echo "* Display setting successful *"

// Net graph
net_graph 1                     // net graph
net_graphpos 1                  // position of net graph {0,1,2}
net_graphproportionalfont 0.8   // font size of net graph

echo "* Net graph setting successful *"

// Volume
voice_enable 1                  // enable voice
voice_mixer_volume "1.0"        // microphone volume
voice_scale "1"                 // teammate voice scaling
voice_modenable "1"
snd_menumusic_volume "0"
snd_deathcamera_volume "0"
snd_mapobjective_volume "0"
snd_roundend_volume "0"
snd_roundstart_volume "0"
snd_tensecondwarning_volume "0.15"
snd_mvp_volume "0"
volume "0.5"

echo "* Sound setting successful *"

// HUD
zoom_sensitivity_ratio_mouse 1  // zoom sensitivity
cl_showloadout 1                // always show weapons and utils
cl_hud_color 10                 // HUD color {0-10}
cl_hud_background_alpha "0.5"   // transparency of HUD
cl_hud_healthammo_style 0
cl_autowepswitch 0              // disable auto weapon switch
hud_scaling 1                   // HUD scaling
mm_dedicated_search_maxping 100 // maximum matching ping
cl_use_opens_buy_menu 0         // use E to buy
closeonbuy "0"

echo "* HUD setting successful *"

// Radar
cl_radar_always_centered 1      // radar centers at player
cl_teammate_colors_show 1
cl_radar_scale "0.3"           // radar scaling
cl_hud_radar_scale "1.2"        // radar HUD scaling
cl_radar_icon_scale_min 0.1     // radar player icon scaling
cl_radar_rotate 0               // disable radar rotation
cl_radar_square_with_scoreboard 1
cl_teamid_overhead_always 1     // show teammate through walls

echo "* Radar setting successful *"

// View model
cl_righthand 1                  // hold weapon with right hand
cl_viewmodel_shift_left_amt "1.5"
cl_viewmodel_shift_right_amt "0.75"
cl_bob_lower_amt 0
cl_bobamt_lat 0
cl_bobamt_lat 0
cl_bobamt_vert 0
cl_bobcycle 0
viewmodel_recoil 0              // weapon recoil scle

viewmodel_fov "60"              // FOV
viewmodel_presetpos 3
viewmodel_offset_x "2"
viewmodel_offset_y "1.5"
viewmodel_offset_z "-1"

echo "* View model setting successful *"
echo "* Generic setting successful *"
```

## Personal Settings

```bash
// Mouse
sensitivity 0.95
m_rawinput 1
m_mouseaccel1 0
m_mouseaccel2 0

echo "* Mouse setting successful *"

// Crosshair
crosshair 1
cl_crosshairstyle "4"
cl_crosshaircolor "4"           // light blue
cl_crosshairsize 4              // crosshair size
cl_crosshairthickness 0.5
cl_crosshairgap -1
cl_crosshairdot 0               // no dot
cl_crosshair_drawoutline 1

echo "* Crosshair setting successful *"

// Key binding
// bind [key] "[command]"
bind MWHEELDOWN "+jump"

alias "+jumpthrow" "+jump;-attack"
alias "-jumpthrow" "-jump"
bind ALT "+jumpthrow"

alias "+cjump" "+jump;+duck"
alias "-cjump" "-jump;-duck"
bind SPACE "+cjump"

bind F1 "buy vest; buy ak47"
bind F2 "buy vesthelm; buy defuser"
bind F3 "buy smokegrenade; buy flashbang; buy hegrenade; buy molotov"
bind F4 "buy smokegrenade; buy flashbang; buy flashbang; buy molotov"
bind F5 "buy awp; buy vest; buy vesthelm" 
bind F7 "toggle cl_crosshairsize 3 999"

echo "* Key binding successful *"
echo "* Personal setting successful *"
```

## One-time execution

To use these configuration in a console instead in a script, copy and paste the console version below.

```
mat_monitorgamma "1.6";sys_antialiasing "0";fps_max 200;mat_savechanges;con_enable 1;cl_autohelp "1";echo "* Display setting successful *";net_graph 1;net_graphpos 1;net_graphproportionalfont 0.8;echo "* Net graph setting successful *";voice_enable 1;voice_mixer_volume "1.0";voice_scale "1";voice_modenable "1";snd_menumusic_volume "0";snd_deathcamera_volume "0";snd_mapobjective_volume "0";snd_roundend_volume "0";snd_roundstart_volume "0";snd_tensecondwarning_volume "0.15";snd_mvp_volume "0";volume "0.5";echo "* Sound setting successful *";zoom_sensitivity_ratio_mouse 1;cl_showloadout 1;cl_hud_color 10;cl_hud_background_alpha "0.5";cl_hud_healthammo_style 0;cl_autowepswitch 0;hud_scaling 1;mm_dedicated_search_maxping 100;cl_use_opens_buy_menu 0;closeonbuy "0";echo "* HUD setting successful *";cl_radar_always_centered 1;cl_teammate_colors_show 1;cl_radar_scale "0.3";cl_hud_radar_scale "1.2";cl_radar_icon_scale_min 0.1;cl_radar_rotate 0;cl_radar_square_with_scoreboard 1;cl_teamid_overhead_always 1;echo "* Radar setting successful *";cl_righthand 1;cl_viewmodel_shift_left_amt "1.5";cl_viewmodel_shift_right_amt "0.75";cl_bob_lower_amt 0;cl_bobamt_lat 0;cl_bobamt_lat 0;cl_bobamt_vert 0;cl_bobcycle 0;viewmodel_recoil 0;viewmodel_fov "60";viewmodel_presetpos 3;viewmodel_offset_x "2";viewmodel_offset_y "1.5";viewmodel_offset_z "-1";echo "* View model setting successful *";echo "* Generic setting successful *";sensitivity 1.15;m_rawinput 1;m_mouseaccel1 0;m_mouseaccel2 0;echo "* Mouse setting successful *";crosshair 1;cl_crosshairstyle "4";cl_crosshaircolor "4";cl_crosshairsize 3;cl_crosshairthickness 0.5;cl_crosshairgap -1.5;cl_crosshairdot 0;cl_crosshair_drawoutline 1;echo "* Crosshair setting successful *";bind MWHEELDOWN "+jump";alias "+jumpthrow" "+jump;-attack";alias "-jumpthrow" "-jump";bind ALT "+jumpthrow";alias "+cjump" "+jump;+duck";alias "-cjump" "-jump;-duck";bind SPACE "+cjump";bind F1 "buy vest; buy ak47";bind F2 "buy vesthelm; buy defuser";bind F3 "buy smokegrenade; buy flashbang; buy hegrenade; buy molotov";bind F4 "buy smokegrenade; buy flashbang; buy flashbang; buy molotov";bind F5 "buy awp; buy vest; buy vesthelm";bind F7 "toggle cl_crosshairsize 3 999";echo "* Key binding successful *";echo "* Personal setting successful *";
```

## Training Script

This script converts a normal BOT match into a map training.

```
sv_cheats 1
mp_autoteambalance 0            // disable team balance
mp_limitteams 0                 // disable team limits
mp_respawn_immunitytime 0       // set immunity time after respawn to 0 
//mp_give_player_c4 0           // no C4
mp_buy_anywhere 1               // able to buy anywhere
mp_maxmoney 50000               // maximum money
mp_startmoney 50000             // start-up money
mp_buytime 999999               // buy time
ammo_grenade_limit_total "6"    // grenade amount
mp_free_armor 0                 // no armor
mp_round_restart_delay 0        // set round restart delay
mp_freezetime 0                 // set freeze time
mp_do_warmup_offine 1           // start offline warmup
mp_do_warmup_period 1           // enable warmup
mp_warmuptime 5400              // set warmup time
sv_infinite_ammo 2              // infinite ammo {0,1,2}
sv_grenade_trajectory 1         // display grenade traces
sv_showimpacts 1                // show hits
mp_forcecamera 0                //
mp_teammates_are_enemies 1      // equal damage to all players
sv_alltalk 1                    // all talk
bot_stop 1                      // bot can't move
bot_join_after_player 1         // bot join after player
mp_restartgame 1                // restart game
bot_kick                        // kick all bots

bind H "noclip"
bind J "bot_place"
bind F6 "god"
//bind "F1" "bot_add_ct"
//bind "F2" "bot_add_t"
```

Reference: 
- [CSGO console commands](https://www.pcgamesn.com/counter-strike-console-commands)
- [Advanced CSGO console command list](https://skins.cash/blog/csgo-console-commands-list/)
- [Steam Community::CS:GO指令大全](https://steamcommunity.com/sharedfiles/filedetails/?id=1228404580)
- [CSGO-config命令详解](https://herixth.top/2019/03/10/CSGO-config%E5%91%BD%E4%BB%A4%E8%AF%A6%E8%A7%A3.html)
