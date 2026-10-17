# Vidyavardhini's College of Engineering & Technology
### Department of Information Technology
**Academic Year 2026–27 — Second Year Engineering (Semester III)**  
**Course:** Applied Mathematics Thinking-I (`2343111`)  
**Curriculum Unit:** Probability & Random Variables  

---

# Library Book Checkout Duration & Delay Risk Analysis via Probability Models

**Project Identification:** Mini Project Group No: 15 &nbsp;|&nbsp; **Batch:** B4  
**Institution:** Vidyavardhini’s College of Engineering & Technology (VCET), Vasai  
**Department:** Department of Information Technology  
**Course Code & Title:** Applied Mathematics Thinking-I (`2343111`)  
**Technology Stack:** Python 3.10+, Streamlit, SymPy, SciPy.stats, NumPy, Pandas, Matplotlib & Seaborn  

---

## 1. Project Title & Aim

### Project Title
**Library Book Checkout Duration & Delay Risk Analysis via Probability Models**

### Aim
To model the discrete count of books borrowed per user and continuous holding/delay times, evaluate raw and central moments, derive Moment Generating Functions (MGF), and assess return delay probabilities to optimize library inventory allocation.

---

## 2. Mathematical Concepts & Curriculum Alignment

### 2.1 Discrete Random Variable ($X$)
Models book checkout counts per student in a single circulation desk visit:
- **Values:** $x \in \{1, 2, 3, 4\}$
- **Probability Mass Function (PMF):**
  $$P(X = x) = \frac{n_x}{N}, \quad \text{where } \sum_{x=1}^4 P(X = x) = 1$$
- **Expected Value (Mean):**
  $$E(X) = \mu_X = \sum_{x=1}^4 x \cdot P(X = x)$$
- **Second Raw Moment:**
  $$E(X^2) = \sum_{x=1}^4 x^2 \cdot P(X = x)$$
- **Variance:**
  $$\sigma_X^2 = \operatorname{Var}(X) = E(X^2) - [E(X)]^2$$

---

### 2.2 Continuous Random Variable ($Y$)
Models book holding time and overdue return delays via continuous probability density functions $f(y)$, calculating mean holding time $E(Y)$ and variance $\sigma_Y^2$:

#### 1. Exponential Distribution ($\operatorname{Exp}(\lambda)$)
$$f(y) = \lambda e^{-\lambda y}, \quad y \ge 0, \; \lambda > 0$$
- Fitted via Maximum Likelihood Estimation (MLE): $\hat{\lambda} = \frac{1}{\bar{y}}$
- Mean: $E(Y) = \frac{1}{\lambda}$
- Variance: $\sigma_Y^2 = \frac{1}{\lambda^2}$

#### 2. Gamma Distribution ($\operatorname{Gamma}(\alpha, \theta)$)
$$f(y) = \frac{1}{\Gamma(\alpha) \theta^\alpha} y^{\alpha - 1} e^{-y / \theta}, \quad y > 0$$
- Fitted via MLE: $\hat{\alpha}$ (shape) and $\hat{\theta} = \frac{1}{\hat{\beta}}$ (scale)
- Mean: $E(Y) = \alpha \theta$
- Variance: $\sigma_Y^2 = \alpha \theta^2$

---

### 2.3 Central Moments, Skewness & Kurtosis
Evaluates distribution asymmetry and tail heavy-tailedness using central moments $\mu_k = E[(Y - E[Y])^k]$ computed directly via `scipy.stats.moment`:

- **1st Central Moment:** $\mu_1 = \text{scipy.stats.moment}(y, 1) \equiv 0$
- **2nd Central Moment:** $\mu_2 = \text{scipy.stats.moment}(y, 2) = \sigma^2$ (Variance)
- **3rd Central Moment:** $\mu_3 = \text{scipy.stats.moment}(y, 3)$ (Asymmetry)
- **4th Central Moment:** $\mu_4 = \text{scipy.stats.moment}(y, 4)$ (Tail heaviness)
- **Karl Pearson's Skewness Coefficient ($\beta_1$):**
  $$\beta_1 = \frac{\mu_3^2}{\mu_2^3}, \qquad \gamma_1 = \frac{\mu_3}{\sigma^3}$$
  - $\gamma_1 > 0 \implies$ Right-skewed delay distribution (most returns are on-time, but rare severe delays drag out the circulation cycle).
- **Kurtosis Coefficient ($\beta_2$) & Excess Kurtosis ($\gamma_2$):**
  $$\beta_2 = \frac{\mu_4}{\mu_2^2}, \qquad \gamma_2 = \beta_2 - 3$$
  - $\beta_2 > 3.0 \implies$ **Leptokurtic** (fat tails), confirming elevated tail risk of severe overdue books compared to a normal distribution.

---

### 2.4 Moment Generating Function (MGF) Engine
Analytically derives the closed-form MGF $M_Y(t) = E[e^{tY}]$ for the continuous delay model and evaluates automated derivatives at $t = 0$:
$$M_Y(t) = E[e^{tY}] = \int_{0}^{\infty} e^{ty} \lambda e^{-\lambda y} \, dy = \lambda \int_{0}^{\infty} e^{-(\lambda - t)y} \, dy$$

For convergence, $\lambda - t > 0 \implies t < \lambda$:
$$M_Y(t) = \frac{\lambda}{\lambda - t} = \left( 1 - \frac{t}{\lambda} \right)^{-1}$$

#### Raw Moments via Differentiation:
$$\mu_k' = \left. \frac{d^k M_Y(t)}{dt^k} \right|_{t=0}$$
1. **1st Derivative (Mean):**
   $$\frac{d M_Y(t)}{dt} = \frac{\lambda}{(\lambda - t)^2} \implies \mu_1' = \left. \frac{\lambda}{(\lambda - t)^2} \right|_{t=0} = \frac{1}{\lambda}$$
2. **2nd Derivative (Second Moment):**
   $$\frac{d^2 M_Y(t)}{dt^2} = \frac{2\lambda}{(\lambda - t)^3} \implies \mu_2' = \left. \frac{2\lambda}{(\lambda - t)^3} \right|_{t=0} = \frac{2}{\lambda^2}$$
3. **Variance Verification:**
   $$\operatorname{Var}(Y) = \mu_2' - (\mu_1')^2 = \frac{2}{\lambda^2} - \left(\frac{1}{\lambda}\right)^2 = \frac{1}{\lambda^2}$$

---

### 2.5 Tail Risk & Custom Loan Period Inventory Allocation
- **Custom Loan Period ($L$):** Predicts overdue rate $P(\text{Holding Time} > L)$ when the library adopts 7, 14, 21, or 28-day borrowing policies.
- **Tail Delay Exceedance:** $P(Y > d) = 1 - F(d)$ for custom threshold days $d$.
- **Automated Risk Categories:**
  - **LOW RISK ($P(Y > d) < 0.15$):** Normal circulation, standard 0%–5% safety buffer.
  - **MODERATE RISK ($0.15 \le P(Y > d) \le 0.40$):** Elevated hold probability, 15%–20% buffer reserve, 2nd tier SMS reminders.
  - **CRITICAL / HIGH RISK ($P(Y > d) > 0.40$):** High stockout probability, urgent recall protocols, escalate fine tiers, procure multi-user digital e-book licenses.

---

## 3. Main Building Blocks of the Python Code

As mandated in Section 4 of the VCET project specification:

| Building Block | Exact Role in Codebase |
| :--- | :--- |
| **`SymPy`** | Performs symbolic evaluation of MGF improper integrals ($\int_0^\infty e^{ty} \lambda e^{-\lambda y} dy$) and executes automated differentiation at $t = 0$ to derive $\mu_1', \mu_2', \operatorname{Var}(Y)$. |
| **`SciPy.stats`** | Executes MLE distribution fitting (`stats.expon.fit`, `stats.gamma.fit`), evaluates CDFs (`stats.expon.cdf`), Kolmogorov-Smirnov goodness-of-fit tests, and **computes central moments via `stats.moment(data, moment=k)`**. |
| **`Matplotlib / Seaborn`** | Renders discrete PMF stem plots, step CDF staircase curves, continuous distribution histograms, empirical KDEs, and fitted probability curves with theme-aware styling (Light & Dark modes). |
| **`Streamlit`** | Builds the production interactive web portal featuring a top-right **Light/Dark Mode toggle button**, parameter tuning, custom loan period adjustments ($L$), delay risk simulations ($d$), and automated metric summary rendering. |
| **`Pandas & NumPy`** | Tabular circulation desk data processing, descriptive summaries, null hygiene audits, and high-performance array operations. |

---

## 4. Application Architecture: Tab-to-PDF Curriculum Mapping

The application is structured into 6 clear, clean tabs without the "Module" keyword, mapped directly to the syllabus sections:

| Tab Name | PDF Curriculum Section Covered | Mathematical & Computational Deliverables |
| :--- | :--- | :--- |
| **Tab 1: Overview & Foundations** | **Section 1 & Section 2**<br>(Project Title, Aim & Mathematical Concepts) | System architecture overview, discrete vs continuous problem formulation, stochastic pipeline diagram, and summary formula reference card. |
| **Tab 2: Dataset Viewer** | **Section 3, Step 1**<br>(Library Data Extraction) | Circulation desk logs with $N = 80$ records ($N \ge 35$), holding days, overdue delay days, fine calculation (₹5/day), CSV export, and null hygiene verification. |
| **Tab 3: Discrete PMF (X)** | **Section 2.1 & Section 3, Step 2**<br>(Discrete Random Variable $X$) | Probability Mass Function table $P(X=x)$ for books borrowed $X \in \{1, 2, 3, 4\}$, Expected Value $E(X)$, Second Moment $E(X^2)$, Variance $\operatorname{Var}(X)$, PMF stem plot, and CDF step staircase plot. |
| **Tab 4: Continuous Delay (Y) & Moments** | **Section 2.2, 2.3 & Section 3, Steps 2, 3**<br>(Continuous Variable $Y$ & Central Moments) | MLE fitting of Exponential & Gamma PDFs, first 4 central moments ($\mu_1=0, \mu_2=\sigma^2, \mu_3, \mu_4$) via `scipy.stats.moment`, Karl Pearson skewness ($\beta_1$), and kurtosis ($\beta_2$, excess $\gamma_2$). |
| **Tab 5: Symbolic MGF Engine** | **Section 2.4 & Section 3, Step 4**<br>(MGF Derivation & Proof via SymPy) | Analytical improper integral $M(t) = \int_0^\infty e^{ty} \lambda e^{-\lambda y} dy = \frac{\lambda}{\lambda - t}$ ($t < \lambda$), automated symbolic differentiation at $t = 0$ for $\mu_1', \mu_2'$, variance proof $\sigma^2 = 1/\lambda^2$, and MGF divergence curve. |
| **Tab 6: Overdue Risk Simulator** | **Section 3, Step 5**<br>(Web Application Deployment & Risk Simulation) | Interactive simulation for **custom loan periods ($L$)**, tail risk exceedance $P(Y > d) = 1 - F(d)$, automated risk classification badges, safety buffer stock recommendations, and Viva defense questions. |

---

## 5. Local Setup & Execution Guide

### Step 1: Navigate to Workspace
```bash
cd AMT-Miniproject
```

### Step 2: (Optional) Activate Virtual Environment
```bash
# Windows:
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux:
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application
```bash
streamlit run app.py
```

Access the live dashboard at:
`http://localhost:8501`

Use the small circular logo button on the **top-right corner of the Institutional Banner** (`☀️` / `🌙`) to toggle between Light and Dark themes.
