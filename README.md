# Leader Detection
This repo uses Cyclicity Analysis to determine the ranking of each FX Pair.
[explain more][to do]
## Cyclicity Analysis of Time-Series
cyclicity_analysis.py is a working implementation of Cyclicity Analysis, which is a pattern recognition technique for analyzing the leader follower dynamics of multiple time-series.

### Introduction

Let $I$ be an interval in $\mathbb R,$ which we regard as a time interval, and let $N \in \mathbb N$. We refer to a smooth (infinitely differentiable map) $\mathbf x: I \rightarrow \mathbb R^N$ is an **($N$-dimensional) time-series/signal/wave/trajectory** on $I.$ Note that we can write $\mathbf x = (x_1 \ , \ \dots \ , \ x_N),$ where each $x_n$ is a one-dimensional time-series on $I,$ which we refer to as the $n$-th **component time-series** of $\mathbf x.$

Given an $N$-dimensional time-series $\mathbf x$ on $I,$ we pose the following questions pertaining to leader follower dynamics of its components.
  * Fix two indices $1 \le m,n \le N$ and consider the component time-series $x_m$ and $x_n.$ Assume that $x_m$ and $x_n$ are approximately the same time-series up to a time lag. Without necessarily knowing the time lag, can we determine whether $x_m$ is leading or following $x_n$ ?
  * Assume that all $N$ component time-series approximately trace a common time-series up to time lags and scaling constants. Without knowing the underlying time-series, can we recovering the order of the time lags ?

Answering these questions is the heart of **Cyclicity Analysis**, a field which arose from Applied Topological Data Analysis that has practical applications in Finance and Neurophysiology. In this notebook, we demonstrate the Cyclicity Analysis pipeline applied to a given $N$-dimensional time-series.

### Step 1 : Determining Pairwise Component Leader Follower Relationships

Throughout, we let $\mathbf x$ be a given $N$-dimensional time-series on an interval $I.$ Our goal is to determine and quantify the pairwise leader follower relationships between any two of its component time-series.

We define the **lead(-lag) matrix** corresponding to $\mathbf x$ to be the $N \times N$ matrix $\mathbf Q$ whose $(m,n)$-th entry is
  \begin{align*}
    Q_{m,n} &= \frac{1}{2} \int_I \mathbf x(t) \  \left(\mathbf x'(t) \right)^T - \mathbf x'(t) \  \left(\mathbf x(t) \right)^T \ dt
  \end{align*}
for all $1 \le m,n \le N.$

The following proposition discusses on why we consider the lead matrix.

**Proposition**: If $\mathbf Q$ is the lead matrix corresponding to $\mathbf x,$ then the following properties hold.
  * For all $1 \le m,n \le N,$ the entry $Q_{m,n}$ is the **oriented (signed) area** enclosed by the $2$-dimensional trajectory on $I$ formed by concatenating the trajectory $(x_m,x_n): I \rightarrow \mathbb R^2$ and the trajectory on $I$ tracing the oriented line segment in $\mathbb R^2$ having endpoints $(x_m(a), x_n(a))$ and $(x_m(b), x_n(b)).$
  * Assume that for some $1 \le m,n \le N,$ there exists a time lag $\tau_{m,n} \in \mathbb R$ such that $x_{n}(t)=x_{m}(t-\tau_{m,n}).$ For sufficiently small $\tau_{m,n},$ the sign of $Q_{m,n}$ equals the sign of $\tau_{m,n}.$
  * $\mathbf Q$ is a skew-symmetric matrix i.e. $\mathbf Q=-\mathbf Q^T.$
  * $\mathbf Q$ has an even rank. Furthermore, if $\lambda \in \mathbb C$ is an eigenvalue of $\mathbf Q$ with corresponding eigenvector $\mathbf v=\left(v_1 \ , \ \dots \ , \ v_N \right) \in \mathbb C^N,$ then $\lambda$ is a purely imaginary value and its complex conjugate $\overline{\lambda}$ is also an eigenvalue of $\mathbf Q$ with corresponding eigenvector $\overline{\mathbf v}=\left(\overline{v_1} \ , \ \dots \ , \ \overline{v_N} \right).$

We refer to the eigenvalue of $\mathbf Q$ with the largest positive imaginary part as the **dominant eigenvalue** and the eigenvector corresponding to this specific eigenvalue as the **leading eigenvector** of $\mathbf Q.$

The second statement of the proposition tells us that the sign of the $(m,n)$-th entry of $\mathbf Q$ captures the essential information between the component time-series $x_m$ and $x_n,$ assuming these time-series are the same up to a small time lag. In particular,
  * $x_m$ leads $x_n$ if $Q_{m,n}>0$
  * $x_m$ follows $x_n$ if $Q_{m,n}<0$

In practice, however, we do not directly deal with the $N$-dimensional time-series $\mathbf x.$ Rather, we observe $\mathbf x$ at $K$ different times, where $K \in \mathbb N$ is a large integer. To this end, if $K \in \mathbb N,$ and if $\left \lbrace t_k \right \rbrace_{k=1}^K$ is a strictly increasing finite sequence of times in $I,$ then we consider the finite set $\left \lbrace \mathbf x_{t_k} \right \rbrace_{k=1}^K,$ in which $\mathbf x_{t_k}=\mathbf x(t_k)$ is the observation of $\mathbf x$ made at the time $t_k$ for each $1 \le k \le K.$ We define the **discrete lead matrix** corresponding to $\left \lbrace \mathbf x_{t_k} \right \rbrace_{k=1}^K$ to be the $N \times N$ matrix
  \begin{align*}
    \widehat{\mathbf Q}_{t_1 \ , \ \dots \ , \  t_K} &= \frac{1}{2} \sum_{k=1}^{K-1} \mathbf x_{t_k} \mathbf x^T_{t_{k+1}} - \mathbf x_{t_{k+1}} \mathbf x^T_{t_{k}}.
  \end{align*}
  

### Example

Fix $N \in \mathbb N$ and $I=[0,1].$ Consider the $N$-dimensional time-series $\mathbf x$ on $I,$ in which $x_n(t)=\sin \left (2 \pi \left(t -\frac{n-1}{N} \right) \right)$ for each $1 \le n \le N.$ One can show that the lead matrix corresponding to $\mathbf x$ has $(m,n)$-th entry equal to $\pi \sin \left( \frac{2 \pi (n-m)}{N}\right)$ for all $1 \le m,n \le N.$ Furthermore, fix $K \in \mathbb N$ and let $t_k=\frac{k-1}{K}$ for each $1 \le k \le K+1.$ One can show the discrete lead matrix corresponding to $\left \lbrace \mathbf x_{t_k} \right \rbrace_{k=1}^{K+1}$ has $(m,n)$-th entry equal to $\frac{K}{2} \sin \left(\frac{2 \pi}{K} \right) \sin \left( \frac{2 \pi (n-m)}{N}\right)$ for all $1 \le m,n \le N.$


We plot the heatmap of the discrete lead matrix when $N=15$ and $K=1000.$

![Lead Lag Matrix](https://github.com/financeckProject/leader-detection/blob/main/docs/lead_lag.png)
## Step 2 : Determining the Sequential Order of Time-Series

We now assume $\mathbf x$ is an $N$-dimensional time-series on $I$ satisfying the **Chain of Offsets Model (COOM)**, in which there is an underlying $P$-periodic map $\phi: \mathbb R \rightarrow \mathbb R$ and offsets $c_1 \ , \ \dots \ , \ c_N>0$ and offsets $\alpha_1 \ , \ \dots \ , \ \alpha_N \in [0,P)$ such that $$x_n(t)=c_n \phi(t-\alpha_n)$$ for each $1 \le n \le N.$ In other words, COOM states that the $N$ component time-series of $\mathbf x$ trace a periodic map up to scaling constants and time lags. Assuming COOM, our goal is to determine the **sequential order** of our $N$ time-series, which is the ordering of the offsets i.e. permutation $\sigma: \left \lbrace 1 \ , \ \dots \ , \ N   \right \rbrace \rightarrow \left \lbrace 1 \ , \ \dots \ , \ N  \right \rbrace$ such that $\alpha_{\sigma(1)} \ \le \ \dots \ \le \  \alpha_{\sigma(N)}.$ The lead matrix will come in handy.

**Proposition**: Let $\phi$ be the $P$-periodic map as defined in COOM. Then, the following properties hold.
  * We have $\phi(t) = \sum_{k \in \mathbb Z} \widehat{\phi}_k e^{\frac{2 \pi i k t}{P}},$ where $\widehat{\phi}_k$ is the $k$-th Fourier Coefficient of $\phi.$
  * The lead matrix $\mathbf Q$ corresponding to $\mathbf x$ under COOM has $(m,n)$-th entry equal to $2 \pi c_m c_n \sum_{k \in \mathbb N} k \left|\widehat{\phi}_k \right|^2 \sin \left(\frac{2 \pi k (\alpha_m-\alpha_n)}{P} \right)$ for all $1 \le m,n \le N.$
  * Assume $\phi$ has one harmonic i.e. there exists exactly one $k \in \mathbb N$ for which $\widehat{\phi}_k \ne 0.$ Then the lead matrix $\mathbf Q$ has rank $2$, and its leading eigenvector is of the form $\mathbf v = e^{i \theta} \left(e^{\frac{2 \pi i \alpha_1}P} \ , \ \dots \ , \ e^{\frac{2 \pi i \alpha_N}P} \right)$ for some $\theta>0.$ Furthermore, the sequential order of our $N$ component time-series is a permutation  $\sigma: \left \lbrace 1 \ , \ \dots \ , \ N   \right \rbrace \rightarrow \left \lbrace 1 \ , \ \dots \ , \ N  \right \rbrace$ such that $\text{Arg} \left(v_{\sigma(1)} \right) \le \ \dots \ \le \text{Arg} \left(v_{\sigma(N)} \right),$ where $v_n$ is the $n$-th component of $\mathbf v$ and $\text{Arg}(v_n) \in [0, 2\pi)$ is the principal argument of $v_n.$

So the last statement tells us that when the periodic map $\phi$ in COOM has only one harmonic, then leading eigenvector of the lead matrix contains information about the offsets. In this case, we can recover the sequential order by simply sorting the components of the leading eigenvector according to their respective principal arguments.

Even if $\phi$ were to have more than one harmonic, nothing changes too much as long as it has a dominating harmonic i.e. there exists exactly one $k \in \mathbb N$ such that $\left|\widehat{\phi}_k \right|^2 \gg \sum_{\ell \ne k} \left|\widehat{\phi}_\ell \right|^2.$ The corresponding matrix term $2 \pi c_m c_n k \left|\widehat{\phi}_k \right|^2 \sin \left(\frac{2 \pi k (\alpha_m-\alpha_n)}{P} \right)$ in the series expansion of the lead matrix $\mathbf Q$ will dominate and serve as a best rank $2$ approximation for the lead matrix $\mathbf Q$ in Frobenius Norm. As a result, the leading eigenvector of $\mathbf Q$ will approximate the leading eigenvector of the dominant matrix term in the series expansion of $\mathbf Q.$

### Example

Reconsider the $N$-dimensional sinusoidal time-series $\mathbf x$ on $[0,1]$ in which $x_n(t)=\sin \left (2 \pi \left(t -\frac{n-1}{N} \right) \right)$ for each $1 \le n \le N.$

On one hand, $\mathbf x$ satisfies COOM with $P=1$ and $\phi(t)=\sin(2 \pi t)$ and $\alpha_n= \frac{n-1}{N}.$  We deduce from the explicit values of the offsets that the sequential order is simply the identity permutation on $\left \lbrace 1 \ , \ \dots \ , \ N \right \rbrace.$

On the other hand, we corroborate the sequential order via the leading eigenvector of the lead matrix $\mathbf Q$ corresponding to $\mathbf x.$ Note $\mathbf Q$ is a rank $2$ circulant skew-symmetric matrix with its $n$-th eigenvector $\mathbf v_n =\frac{1}{\sqrt N} \left(1 , e^{\frac{2 \pi i n}{N}} \ , \ \dots \ , e^{\frac{2 \pi i (N-1)}{N}} \right)$ corresponding to its $n$-th eigenvalue $\lambda_n = \pi \sum_{p=1}^N \sin \left(\frac{2 \pi(p-1)}{N} \right) e^{\frac{2 \pi i n (p-1)}{N}}$ for each $1 \le n \le N.$ Furthermore, the dominant eigenvalue is $\lambda_1=\frac{N \pi i}{2}$ and leading eigenvector is $\mathbf v_1.$ The principal argument of the $p$-th component of $\mathbf v_1$ is $\frac{2 \pi (p-1)}{N},$ and so we see the components of $\mathbf v_1$ are already sorted according to their principal arguments. Hence, the sequential order is simply the identity permutation on $\left \lbrace 1 \ , \ \dots \ , \ N \right \rbrace.$

We plot the components (as coordinates in $\mathbb R^2$) of the leading eigenvector corresponding to the discrete lead matrix from before. We also plot the moduli of the $N$ eigenvalues of the discrete lead matrix sorted in descending order.

![Leading Eigenvector](https://github.com/financeckProject/leader-detection/blob/main/docs/leading_eigenvector.png)

### Application : Stock Price Analysis

As a practical example, we examine the historical time-series of daily stock (closing) price time-series for $N$ given number of companies in the financial sector.

We collect such data from Yahoo Finance below. Following the standard preprocessing procedures with such time-series, we take the logarithm, detrend, and normalize each stock price time-series, and translate each resulting stock price time-series by the first day stock-price value. To this end, we let $\left \lbrace x_{n, k} \right \rbrace_{k=1}^K$ be the stock price time-series of the $n$-th company, where $x_{n, k}$ is the preprocessed closing price of the $n$-th company at day $k,$ according to the above procedure. We note $k=1$ corresponds to first trading day of $2019$, while $K$ corresponds to the latest, most recent trading day before today (at the time of writin).

We plot our $N$ preprocessed time-series and the corresponding discrete lead matrix.

![Lead Lag Matrix](https://github.com/financeckProject/leader-detection/blob/main/docs/lead_lag_matrix.png)

We also plot the leading eigenvector components and the lead matrix eigenvalue moduli sorted in descending order.

![leading eigenvector](https://github.com/financeckProject/leader-detection/blob/main/docs/leading_eigenvector_2.png)

### Accumulated Oriented Areas

In practical applications, given an $N$-dimensional time-series $\mathbf x$ on $I=[a,b]$, it is useful to look at the matrix-valued function $s \mapsto \mathbf Q(s),$ where $\mathbf Q(s)$ is the lead matrix corresponding to $\mathbf x$ restricted to $[a,s]$ for each $s \in I.$ In particular, for fixed $1 \le m,n \le N,$ the map $s \mapsto Q_{m,n}(s)$ is the **accumulated oriented area** corresponding to the component time-series $x_m$ and $x_n.$ The accumulated oriented area reveals the evolution leader follower relationship between $x_m$ and $x_n$ throughout time. We plot several accumulated oriented areas.

![ms price](https://github.com/financeckProject/leader-detection/blob/main/docs/ms_price.png)
![gs](https://github.com/financeckProject/leader-detection/blob/main/docs/gs.png)
![hdb](https://github.com/financeckProject/leader-detection/blob/main/docs/hdb.png)
![ibkr](https://github.com/financeckProject/leader-detection/blob/main/docs/ibkr.png)


### Usage

```python
from cyclicity_analysis import OrientedArea, COOM

df = pd.DataFrame([[0, 1], [1, 0], [0, 0]], columns=['0', '1'])


oa = OrientedArea(df)
# Returns the lead lag matrix of df as a dataframe
lead_lag_df = oa.compute_lead_lag_df()

coom = COOM(lead_lag_df)
# Returns leading eigenvector of lead lag matrix as a numpy array
leading_eigenvector = coom.get_leading_eigenvector()
lead_lag_df , leading_eigenvector
 ```
# Requirements
Download [Python >=3.7](https://www.python.org/downloads/)

# Installation

```bash
python -m pip install -r requirements.txt
```
# Running the Code
[to do]
