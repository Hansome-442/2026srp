# 代码协作说明（GitHub）

> 仓库地址：**https://github.com/Hansome-442/2026srp**
> 最后更新：2026-11-17

---

## 一、先加入仓库

仓库是 **Private**（私有），必须先被邀请才能访问。

**管理员做**（只有负责人需要）：

1. 打开 <https://github.com/Hansome-442/2026srp>
2. 右上角 **Settings** → 左侧 **Collaborators**
3. 点 **Add people**，输入成员的 GitHub 用户名
4. 成员收到邮件，点 **Accept invitation**

> GitHub 免费版的私有仓库**协作者人数不限**，7 个人没问题。

---

## 二、把代码拿到本地

### 2.1 装 Git

先确认你有没有：

```bash
git --version
```

没有的话去 <https://git-scm.com/download/win> 下 Windows 版，一路默认安装。

### 2.2 配置身份（只需一次）

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

### 2.3 克隆

```bash
cd /d D:\
git clone https://github.com/Hansome-442/2026srp.git
cd 2026srp
```

第一次会要求登录，两种方式：

| 方式 | 说明 |
|---|---|
| **浏览器登录**（推荐） | 弹出 GitHub 登录窗口，点 Authorize，会自动存凭据 |
| **Personal Access Token** | Settings → Developer settings → Tokens，生成后当密码用 |

> 用 SSH 也可以，但要先生成密钥并加到 GitHub，多三步。先用 HTTPS 最省事。

---

## 三、日常流程（重要，别直接在 main 上改）

```bash
# 1. 每次开工前先同步
git pull

# 2. 建自己的分支
git checkout -b feature/你负责的部分

# 3. 改代码……

# 4. 提交
git add -A
git commit -m "feat: 说明你改了什么、为什么"

# 5. 推上去
git push -u origin feature/你负责的部分

# 6. 到 GitHub 页面点 Compare & pull request，等合并
```

**分支命名习惯**：

- `feature/xxx` — 新功能
- `fix/xxx` — 修 bug
- `docs/xxx` — 只改文档

---

## 四、绝对不能提交的东西

`.gitignore` 已经挡住了，但你要知道为什么：

| 别提交 | 原因 |
|---|---|
| `data/` | 5.3 GB，仓库会炸 |
| `runs/`、`*.pt` | 训练产物和权重，走网盘 |
| `*.mp4` | 演示视频体积大 |
| `*.zip` | 交接包、数据包 |
| `notes/sessions/` | 本地会话档案 |

**如果误提交了大文件**，不要直接再提交一次删除（历史里还留着），联系负责人处理。

---

## 五、网络不好怎么办

GitHub 在国内时通时断，这里有三档方案，从简单到麻烦：

### 档 1：多试几次

`git clone` 失败就重跑一次。DNS 抖动导致同一命令有时通有时不通，重试往往就好了。

### 档 2：SSH over 443

如果 22 端口被墙（报 `Connection timed out`），改用 443 端口。编辑 `C:\Users\<你>\.ssh\config`：

```
Host github.com
  HostName ssh.github.com
  Port 443
  User git
```

然后把 clone 地址换成 SSH 格式：

```bash
git clone git@github.com:Hansome-442/2026srp.git
```

### 档 3：完全连不上 —— 用 bundle

负责人会提供一个 `2026srp_20261117.bundle`（1.2 MB，群文件里）：

```bash
git clone 2026srp_20261117.bundle 2026srp
cd 2026srp
```

这样你得到一个**完整的 git 仓库**（带历史、能提交），只是暂时推不上 GitHub。
等网络好了，加个远程：

```bash
git remote add origin https://github.com/Hansome-442/2026srp.git
git push -u origin main
```

---

## 六、常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| `Repository not found` | 还没被邀请，或没接受邀请 | 找负责人加 collaborator |
| `Permission denied (publickey)` | SSH 密钥没配 | 换 HTTPS 地址克隆 |
| `fatal: not a git repository` | 不在项目目录里 | `cd` 到仓库根再执行 |
| `Authentication failed` | 密码方式已废弃 | 用浏览器登录或 PAT |
| `error: Your local changes would be overwritten` | 本地有未提交改动 | 先 `git stash`，`git pull`，再 `git stash pop` |
| 推送很慢/超时 | 网络抖动 | 重试；或改用档 2 / 档 3 |

---

## 七、提 PR 的时候写什么

一句人话就够，但要说清三件事：

```
改了什么：给预警加了连续 3 帧确认
为什么：单帧误检会导致趴桌误报，实测误报率约 20%
怎么验证：在 test_images 里挑 5 段视频跑了对比
```

**别写**「update」「fix bug」「修改」这种——三个月后没人知道改了什么。

---

## 八、数据集怎么同步（三台机器训练必读）

代码走 git 很自然，但**数据集 5.3 GB 不进 git**，所以它最容易出问题：
你补了 200 张图、他那边还是旧的，两个人各训一版，指标根本没法比。

### 用「版本号 + 指纹」判断是不是同一份数据

```bash
python scripts/dataset_fingerprint.py          # 扫描并写入 data/DATASET_VERSION.txt
python scripts/dataset_fingerprint.py --check  # 只比对，不一致返回退出码 1
```

会得到这样一份记录：

```
version    : v2
images     : 46960
labels     : 46960
fingerprint: 8f3c2a1d9e
updated    : 2026-11-25 14:20
```

**指纹（fingerprint）由每张图的文件名+大小、每个标签的文件名+大小聚合而成**，
少一张图、改一个标签，指纹就变。扫描 4.6 万张图约 **20 秒**。

### 训练时会自动打印

```
INFO 数据集版本 v2 ｜ 图片 46960 ｜ 指纹 8f3c2a1d9e ｜ 更新于 2026-11-25 14:20
INFO （三台机器训练时请核对指纹一致，否则指标不可比）
```

三个人训练前**对一下这行指纹**，一致才开工。

### 数据变动流程

```
① 新样本统一放 data/incoming/          （不要直接改 data/processed）
② 负责人合并 → python scripts/dataset_fingerprint.py
   → 版本自动从 v2 升到 v3
③ 重新打包传网盘：python scripts/pack_dataset.py
④ 群里发一句「数据集升到 v3，指纹 8f3c2a1d9e，请重新下载」
⑤ 大家删掉旧 data/processed，解压新的
```

**为什么只让一个人改 data/processed**：三个人各自改，合并时会变成三份互不兼容的数据，
谁都说不清最终版本是什么。**数据合库是集中动作，代码才是分布式的。**

### 训练报告里必须写版本

```markdown
数据：v3（指纹 8f3c2a1d9e，46,960 张）
代码：commit 7a3f2c1
结果：mAP50 0.58
```

缺了这两行，别人无法复现你的结果。
