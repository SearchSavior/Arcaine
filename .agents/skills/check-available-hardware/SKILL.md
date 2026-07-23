---
name: check-available-hardware
description: >
  Query available hardware and obtain details about the device architecture to inform hardware aware implementations.
---

## What it does

Check available resources available on the current host, and get a report on their SYCL specifc hardware features, critical facts for hardware optimization tasks.


## How to use

```
./get_device_props
```

## Restrictions

- Do not run on the host
- Make sure you are in the container to run this at all

## Always

- Stop and ask the user what to do if you can't satisfy the restrictions.

