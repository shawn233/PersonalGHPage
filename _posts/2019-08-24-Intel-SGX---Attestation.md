---
layout:     post
title:      Intel SGX | Attestation
subtitle:   attestation
date:       2019-08-24
author:     Xinyu Wang
header-img: img/post-bg-cook.jpg
catalog:    true
tags:
    - Intel SGX
---

This post is simply a note of the [official user guide](https://software.intel.com/en-us/node/702982). Refer to the original page for more details.

---

The purpose of attestation is to demonstrate a piece of software has been established on a platform.

Attestation is suggested to apply prior to provisioning that software with secrets and protected data.

Intel SGX supports two types of attestation, which will be introduced, namely, Local Attestation and Remote Attestation.

## Local Attestation

In local attestation, two applications on the same platform authenticate each other.

The process of local attestation is based on a symmetric key system. The hardware feature that enables the local attestation is that the key is embedded in the hardware platform so it ensures the key can not be tampered with.  

![Local Attestation](https://upload-images.jianshu.io/upload_images/10549717-8261d9f67725386b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Refer to the illustration of local attestation, the process of application B confirms application A is on the same platform can be described as below. After each step, I will add my comment of the process, which will help understand the whole process.  

1. After applications A and B have established a communication path on the untrusted user platform, enclave B sends its MRENCLAVE identity to enclave A. *Comment: According to hardware feature, what application B needs to confirm is that enclave A can produce a report that is encrypted with the exact key on B's hardware platform. The MRENCLAVE identity sent to A is treated as a challenge to provide replay protection, which can be randomized.*

2. After application A receives B's MRENCLAVE identity, it asks the hardware to produce a report containing enclave B's identity in it, and then send it to B. *Comment: By containing B's identity, enclave A resolves the challenge from B. This step provides premise for mutual authentication and even secure transmission. For mutual authentication, enclave A only needs to sends its MRENCLAVE identity to B as a challenge, and requests B to send back a report encrypted with the same key. For secure transmission, enclave A can apply a Diffie-Hellman key exchange protocol, by including a part of the secret key in this report, and waits for B's report to extract the other part and combines them to get the complete key.*

3. After application B receives A's report, enclave B asks the hardware to verify the report to affirm that enclave A is on the same platform as enclave B. B can complete the process of mutual authentication by generating and sending a report containing A's identity.

4. Enclave A then verifies B's report to affirm B is on the same platform.

Reference: [Local attestation by SSLab in GeorgiaTech](https://gts3.org/pages/local-attestation.html)


## Remote Attestation

An application that hosts an enclave can also ask the enclave to produce a report and then pass this report to a platform service to produce a type of credential that reflects the enclave and platform state. This credential is known as quote. This quote can then be passed to entities off of the platform, and verified using Intel<sup>®</sup> Enhanced Privacy ID (Intel<sup>®</sup> EPID) signature verification techniques. As a result, the CPU key is never directly exposed outside the platform.

