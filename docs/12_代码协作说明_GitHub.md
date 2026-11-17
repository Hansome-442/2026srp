# GitHub 使用手册（零基础版）

> 假设你**从没用过 Git**。照着做，每一步都写了「敲什么、看到什么」。
> 最后更新：2026-11-17

---

## 一、先装 Git（只做一次）

打开 <https://git-scm.com/download/win>，下载 **64-bit Git for Windows Setup**，一路 Next 装完。

装完打开 **Anaconda Prompt**，敲：

```bash
git --version
```

看到 `git version 2.x.x` 就成功了。

---

## 二、告诉 Git 你是谁（只做一次）

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

这个名字会显示在每次提交上，用真名或常用昵称。

---

## 三、把代码拿到自己电脑（只做一次）

```bash
cd /d D:\
git clone https://github.com/Hansome-442/2026srp.git
cd 2026srp
```

会弹窗要求登录 GitHub，**选浏览器登录**（Sign in with your browser），点 Authorize。

✅ 完成后 `D:\2026srp` 里就是整个项目。

> **如果交接包已经解压到 `D:\2026srp`**，就不用 clone 了。
> 在那目录里敲 `git status`，不报 `not a git repository` 就说明它已经是仓库。

---

## 四、每天开工的第一件事

```bash
cd /d D:\2026srp
git checkout main        # 切到主线
git pull                 # 拉取别人的最新改动
```

**每次动手前都要做这两步**，否则你是在旧代码上改。

---

## 五、开一个自己的分支

```bash
git checkout -b feature/你的英文名-做什么
```

例子：

```bash
git checkout -b feature/wang-alert-fix
git checkout -b feature/li-ui-fix
```

**命名规则**：全小写英文 + 连字符，别用中文。

确认自己在哪：

```bash
git branch
```

带 `*` 的就是当前分支：

```
  main
* feature/wang-alert-fix
```

---

## 六、改代码，然后提交

改完先看自己动了哪些文件：

```bash
git status
```

然后：

```bash
git add -A
git commit -m "feat: 说明你改了什么"
```

**提交信息的写法**：

| ✅ 好 | ❌ 差 |
|---|---|
| `feat: 预警改成连续3帧确认` | `update` |
| `fix: 修复摄像头关闭后画面残留` | `改了一下` |
| `docs: 补充路径排查说明` | `123` |

前缀：`feat` 新功能 / `fix` 修 bug / `docs` 只改文档 / `chore` 杂事。

---

## 七、推送到 GitHub

```bash
git push -u origin feature/wang-alert-fix
```

第一次推要带 `-u origin 分支名`，以后只敲 `git push` 就行。

推成功后 GitHub 上**自动出现**这个分支 —— 不用你手动建。

---

## 八、开 Pull Request（在网页上）

1. 打开 <https://github.com/Hansome-442/2026srp>
2. 页面顶部会出现一条黄色横幅：

```
feature/wang-alert-fix had recent pushes — [Compare & pull request]
```

3. 点那个绿色按钮
4. 填两项：

| 字段 | 填什么 |
|---|---|
| Title | 一句话，如 `预警改成连续3帧确认` |
| Description | 改了什么 / 为什么改 / 怎么验证的 |

5. 点绿色的 **Create pull request**

✅ 之后负责人的 GitHub 会收到通知，审核后合并。

---

## 九、合并之后

负责人合并了你的代码，你要：

```bash
git checkout main
git pull
```

本地 main 才跟上。**否则你一直在旧代码上改。**

---

## 十、常见报错对照

| 报错 | 原因 | 解决 |
|---|---|---|
| `fatal: not a git repository` | 不在项目目录 | `cd /d D:\2026srp` |
| `Permission denied (publickey)` | SSH 密钥没配 | 改用 https 地址 clone |
| `Authentication failed` | 密码方式已废弃 | 浏览器登录，或用 Personal Access Token 当密码 |
| `Updates were rejected` | 远程有你没有的提交 | 先 `git pull`，再 push |
| `Your local changes would be overwritten` | 本地有未提交改动 | `git stash` → `git pull` → `git stash pop` |
| `GH006: Protected branch update failed` | 你直接推了 main | **正常！** 必须走分支 + PR |
| `git: command not found` | 没装 Git 或用错终端 | 装 Git，用 Anaconda Prompt |

---

## 十一、三种“后悔药”

| 情况 | 命令 |
|---|---|
| 改了没提交，想全撤销 | `git checkout .` |
| 提交了想撤销这次提交（改动保留） | `git reset --soft HEAD~1` |
| 只想还原某个文件 | `git checkout -- 文件名` |

⚠️ `git checkout .` 会**永久丢掉**未提交的改动，用之前想清楚。

---

## 十二、命令行实在搞不懂：用 GitHub Desktop

图形界面，鼠标点点就行，功能完全一样。

1. 下载 <https://desktop.github.com/>
2. 登录 GitHub 账号
3. `File → Clone repository` → 选 `Hansome-442/2026srp`
4. 之后：
   - 左上角 **Current Branch** → `New Branch` 建分支
   - 左下角填 Summary → 点 **Commit to xxx**
   - 顶部 **Push origin** 上传
   - 网页上开 PR（同第八节）

**推荐给完全不想碰命令行的人。** 但命令行更通用，学会一次受益很久。

---

## 十三、一张图记住全流程

```
git checkout main
git pull                        ← 拿最新
     ↓
git checkout -b feature/xxx     ← 开分支
     ↓
改代码
     ↓
git add -A
git commit -m "feat: ..."       ← 存档
     ↓
git push -u origin feature/xxx  ← 上传分支
     ↓
网页点 Create pull request      ← 请求合并
     ↓
负责人审核 → Merge
     ↓
git checkout main
git pull                        ← 拿到合并后的结果
```

---

## 十四、五条铁律

1. **永远不要直接在 main 上改** —— 推不上去（仓库开了分支保护）
2. **每次开工前先 `git pull`** —— 否则你改的是旧代码
3. **提交信息要说人话** —— 三个月后自己都看不懂 `update`
4. **不确定就先 `git status`** —— 它永远告诉你现在的状态
5. **搞砸了别慌** —— 代码在 GitHub 上有一份，删掉重新 clone 就行

---

## 附：数据集怎么同步（训练的人必读）

代码走 git 很自然，但**数据集 5.3 GB 不进 git**，所以它最容易出问题。

### 判断“是不是同一份数据”看指纹

```bash
python scripts/dataset_fingerprint.py --check
```

- `一致 ✅` → 和记录的版本一样，可以开工
- `不一致 ❗` → 数据被改过，先问负责人

### 训练时会自动打印

```
INFO 数据集版本 v2 ｜ 图片 46960 ｜ 指纹 8f3c2a1d9e
INFO （三台机器训练时请核对指纹一致，否则指标不可比）
```

**几个人同时训练前，对一下这行指纹。**

### 数据由负责人统一合并

```
① 新样本放 data/incoming/（不要直接改 data/processed）
② 负责人：merge_new_data.py → dataset_fingerprint.py → pack_dataset.py --only-new
③ 群里发「数据集升到 v3，请下载增量包」
```

**为什么只让一个人改数据**：三个人各自改，会变成三份互不兼容的数据，谁都说不清最终版本。

### 训练报告里必须写两行

```markdown
数据：v3（指纹 8f3c2a1d9e，46,960 张）
代码：commit 7a3f2c1
```

缺了这两行，别人无法复现你的结果。
