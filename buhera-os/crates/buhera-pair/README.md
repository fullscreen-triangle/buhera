# buhera-pair

The client CLI for a [Buhera](https://github.com/fullscreen-triangle/buhera)
gateway account. Install this on the machine you want to compute on — your
own workstation, not a server — and it holds a credential minted on the web,
then does the actual work the gateway routes to it.

## Pair this machine

1. On the web, sign in and open **pair a machine**. Name it and copy the
   `buhera-pair pair ...` command shown — it includes a one-time token,
   visible only once.
2. Run that command here:

   ```sh
   buhera-pair pair --token <token> --name <name> --gateway <gateway-url>
   ```

## Start accepting work

```sh
buhera-pair run
```

Holds a connection open to the gateway and executes whatever vaHera work
gets dispatched to this machine, against a local kernel. Runs in the
foreground — stop it with Ctrl-C. Reconnects automatically if the network
drops.

## Other commands

```sh
buhera-pair status   # confirm the stored pairing still works
buhera-pair forget   # clear the local credential (does not unpair on the gateway)
```

## Why a separate binary from the gateway

The gateway is a server, deployed once, reached by many accounts. This is
the opposite: one small program, installed by each person pairing their own
machine, that does nothing until you run `pair` and `run`. It never listens
for inbound connections — it dials out to the gateway, which is what lets it
work from behind a home router or office firewall with no port forwarding.
