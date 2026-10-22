#!/usr/bin/env python3
"""Configuration of the private Milvus standalone used by bench/run_milvus.py (Milvus 2.6.18 from the official .deb,
extracted without root into third_party/milvus/root by scripts/setup_milvus.sh).
  python3 bench/milvus_conf.py        -> data/milvus/configs/{milvus.yaml, embedEtcd.yaml, glog.conf}, data/milvus/root_password
Starting from the package's own configuration it changes only what a shared machine needs: every data, log, and
cache path lives under data/milvus; embedded etcd and every component listen on 127.0.0.1; authentication is on with
a random root password (file mode 600); the HTTP/RESTful server, WebUI, and pprof are off. Index and search
behaviour are left at the package defaults."""
import os, re, secrets, shutil

PKG = 'third_party/milvus/root/etc/milvus/configs'; OUT = 'data/milvus/configs'; VAR = os.path.abspath('data/milvus/var')

def sub(s, old, new, count=1):
    assert s.count(old) >= 1, old
    return s.replace(old, new) if count == 0 else s.replace(old, new, count)

def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(VAR, exist_ok=True)
    pw_file = 'data/milvus/root_password'
    if not os.path.exists(pw_file):
        fd = os.open(pw_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as f: f.write(secrets.token_urlsafe(24))
    pw = open(pw_file).read().strip()
    s = open(f'{PKG}/milvus.yaml').read()
    s = sub(s, 'data.dir: /var/lib/milvus/etcd', f'data.dir: {VAR}/etcd')
    s = sub(s, 'path: /etc/milvus/configs/embedEtcd.yaml', f'path: {os.path.abspath(OUT)}/embedEtcd.yaml')
    s = sub(s, 'path: /var/lib/milvus/data/', f'path: {VAR}/data/')
    s = sub(s, 'path: /var/lib/milvus/rdb_data', f'path: {VAR}/rdb_data')
    s = sub(s, 'localPath: /tmp/milvus_access', f'localPath: {VAR}/access')
    s = sub(s, 'local_resource_path: /var/lib/milvus/analyzer', f'local_resource_path: {VAR}/analyzer')
    s = sub(s, '    rootPath: \n    maxSize: 300', f'    rootPath: {VAR}/logs\n    maxSize: 300')
    s = sub(s, '  stdout: true # Stdout enable or not', '  stdout: false # Stdout enable or not')
    s = re.sub(r'(?m)^  ip:  # TCP/IP address of', '  ip: 127.0.0.1 # TCP/IP address of', s)
    s = sub(s, 'enabled: true # Whether to enable the http server', 'enabled: false # Whether to enable the http server')
    s = sub(s, 'enablePprof: true', 'enablePprof: false'); s = sub(s, 'enableWebUI: true', 'enableWebUI: false')
    s = sub(s, 'authorizationEnabled: false', 'authorizationEnabled: true')
    s = sub(s, 'defaultRootPassword: Milvus', f'defaultRootPassword: "{pw}"')
    fd = os.open(f'{OUT}/milvus.yaml', os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f: f.write(s)
    e = open(f'{PKG}/embedEtcd.yaml').read().replace('http://0.0.0.0:2379', 'http://127.0.0.1:2379') + f'data-dir: {VAR}/etcd\n'
    open(f'{OUT}/embedEtcd.yaml', 'w').write(e)
    shutil.copy(f'{PKG}/glog.conf', f'{OUT}/glog.conf')
    if os.path.isdir(f'{PKG}/advanced'): shutil.copytree(f'{PKG}/advanced', f'{OUT}/advanced', dirs_exist_ok=True)
    print('milvus config written to', OUT)

if __name__ == '__main__': main()
