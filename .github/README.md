# Kubernetes

## Viewing Optuna dashboard

Run port forwarding:

```sh
kubectl -n <namespace> port-forward pod/<pod name> 8080:8080
```

Then open `http://127.0.0.1:8080/dashboard/` in browser.

If you have run the k8s in SSH server, create a connection:

```sh
ssh -L 8080:127.0.0.1:8080 <user>@<ssh-server>
```

Then open `http://127.0.0.1:8080/dashboard/` in your local machine.
