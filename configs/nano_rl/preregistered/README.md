# Preregistered RL Configs

Add a `preregistration` mapping only when a run has an immutable study plan.
There is no separate execution mode: every queue item receives the same source,
environment, launch-contract, guard, and evaluation checks. The mapping adds a
hash-pinned registration, content-family artifacts, protocol-matched SFT
baseline, validation-only selection policy, registered seed, endpoints, and
checkpoint schedule.

Test evaluation remains a separate one-shot stage after checkpoint selection.
It must not appear in a training or HPO queue.

Each registered seed is a separate queue item. One byte-identical relaunch is
allowed only before the first optimizer update; a later failure consumes that
registration. The current R33 confirmatory plan requires two seeds and archives
its validation-family power calculation before launch. The queue also requires
hash-pinned passing family-seal and kernel-compatibility reports; confirmatory
items require the clustered-power report from `scripts/nano_clustered_power.py`.
