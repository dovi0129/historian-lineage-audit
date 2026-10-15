"""cstr_dynamic.py — B2용 동적 CSTR: 진짜 플랜트 · 모델 클래스 · 레짐 앵커 · 제어기 패널.

배경: 310/311은 CSTR에서 프로토콜 Step 1–4(진단·라우팅·앵커·앙상블 분산)를 **정상상태 맵**
수준에서 시연했고, 논문 \\ref{sec:cstr}은 "Full controller-panel scoring is left to future work"
라고 자인했다. 이 모듈은 그 구멍을 메우기 위해 동적 환경과 제어기 패널을 제공한다.

플랜트(Henson--Seborg 표준 CSTR, 311과 동일 파라미터):
    dC_A/dt = (q/V)(C_Af - C_A) - k(T) C_A^n
    dT/dt   = (q/V)(T_f - T) + (-dH/rho Cp) k(T) C_A^n + (UA/rho Cp V)(T_c - T) + r
    k(T)    = k0 exp(-E/(R T))
조작변수 = 냉각재 온도 T_c, 제어대상 = 반응기 온도 T.

식별 구조(310/311의 진단 결과를 그대로 승계):
  E      비식별(폐루프에서 T가 좁은 대역에 묶여 온도민감도가 여기되지 않음) → **fix**(Arrhenius)
  kref   식별 가능한 rate 레벨                                              → **fit**
  r_g    레짐별 촉매활성에서 오는 운전 레벨                                  → **anchor**
  n      진짜 반응차수 n*=1.15 vs 모델 n=1 — 구조 mismatch(합성 컴패니언의 kappa*에 대응)

논문 본문과의 대응: r은 drying의 u_off(가법 레벨 앵커)와 같은 역할이며, 모델의 정상상태
열수지 잔차로 적합된다 — 즉 '레벨만 옮기고 응답 기울기는 건드리지 않는' 동일한 성질.
"""
import numpy as np

# ── 진짜 플랜트 상수 (311과 동일·Henson–Seborg) ──
TRUE = dict(qV=1.0, CAf=1.0, Tf=350.0, k0=7.2e10, ER=8750.0,
            mdH=5.0e4 / 239.0, UAcoup=6.5, nstar=1.15)
TREF = 350.0
KREF_TRUE = TRUE["k0"] * np.exp(-TRUE["ER"] / TREF)

# 모델 클래스 nominal (E는 fix·kref는 fit·n=1)
NOMINAL = dict(qV=TRUE["qV"], CAf=TRUE["CAf"], Tf=TRUE["Tf"], mdH=TRUE["mdH"],
               UAcoup=TRUE["UAcoup"], ER=TRUE["ER"], kref=KREF_TRUE, n=1.0, r=0.0)

# 운전점(311과 동일): T_sp≈348.5–351.5 K → T_c≈333.8–335.6 K. 이 대역에서 개루프 안정
# (선형화 최대 실수부 고유값 ≈ −0.8 /min, 시상수 ≈ 1.2 min)이며 n=1 vs n*=1.15의 구조
# mismatch가 T_c 기준 −0.7 ~ −0.9 K로 나타난다 — 앵커가 흡수해야 할 레벨이다.
TC_LO, TC_HI = 325.0, 345.0     # 냉각재 온도 가동범위
TC_RATE = 1.0                   # 스텝당 최대 변화 (K)
DT = 0.1                        # 제어 주기 (min)
N_SUB = 10                      # 적분 서브스텝
SETPOINTS = (348.5, 350.0, 351.5)


def k_of(T, kref, ER):
    """Arrhenius를 T_REF 기준으로 정규화한 형태 — kref(레벨)와 ER(형상)이 분리된다."""
    return kref * np.exp(-ER * (1.0 / np.maximum(T, 1.0) - 1.0 / TREF))


def deriv_true(CA, T, Tc, a=1.0, p=TRUE):
    """진짜 플랜트의 도함수. a = 레짐 촉매활성 배율(운전 레벨의 원천)."""
    kk = k_of(T, KREF_TRUE * a, p["ER"])
    rate = kk * max(CA, 0.0) ** p["nstar"]
    dCA = p["qV"] * (p["CAf"] - CA) - rate
    dT = p["qV"] * (p["Tf"] - T) + p["mdH"] * rate + p["UAcoup"] * (Tc - T)
    return dCA, dT


def deriv_model(CA, T, Tc, c):
    """모델 클래스의 도함수(n=1·가법 레벨 앵커 r 포함)."""
    kk = k_of(T, c["kref"], c["ER"])
    rate = kk * max(CA, 0.0) ** c["n"]
    dCA = c["qV"] * (c["CAf"] - CA) - rate
    dT = c["qV"] * (c["Tf"] - T) + c["mdH"] * rate + c["UAcoup"] * (Tc - T) + c["r"]
    return dCA, dT


def _rk4(f, CA, T, Tc, dt):
    k1 = f(CA, T, Tc)
    k2 = f(CA + 0.5 * dt * k1[0], T + 0.5 * dt * k1[1], Tc)
    k3 = f(CA + 0.5 * dt * k2[0], T + 0.5 * dt * k2[1], Tc)
    k4 = f(CA + dt * k3[0], T + dt * k3[1], Tc)
    return (CA + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            T + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))


def advance(CA, T, Tc, f, dt=DT, n_sub=N_SUB):
    h = dt / n_sub
    for _ in range(n_sub):
        CA, T = _rk4(f, CA, T, Tc, h)
        CA = float(np.clip(CA, 1e-9, 10.0))
        T = float(np.clip(T, 250.0, 500.0))
    return CA, T


def steady_true(Tc, a=1.0, p=TRUE, iters=400):
    """주어진 T_c에서 진짜 플랜트의 정상상태 (적분으로 수렴)."""
    CA, T = 0.5, 350.0
    for _ in range(iters):
        CA, T = advance(CA, T, Tc, lambda c, t, u: deriv_true(c, t, u, a, p), dt=0.5, n_sub=5)
    return CA, T


def steady_model(Tc, c, iters=400):
    CA, T = 0.5, 350.0
    for _ in range(iters):
        CA, T = advance(CA, T, Tc, lambda x, t, u: deriv_model(x, t, u, c), dt=0.5, n_sub=5)
    return CA, T


def tc_for_setpoint(T_sp, plant_fn, lo=TC_LO, hi=TC_HI, iters=40):
    """정상상태 T가 T_sp가 되는 T_c를 이분법으로 찾는다. plant_fn(Tc) -> (CA, T)."""
    flo = plant_fn(lo)[1] - T_sp
    fhi = plant_fn(hi)[1] - T_sp
    if flo * fhi > 0:
        return lo if abs(flo) < abs(fhi) else hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if (plant_fn(mid)[1] - T_sp) * flo > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def anchor_residual(T, Tc, CA, c):
    """정상상태 열수지 잔차 = 모델이 놓친 레벨. drying의 정적 잔차 r과 같은 정의."""
    kk = k_of(T, c["kref"], c["ER"])
    rate = kk * max(CA, 0.0) ** c["n"]
    return -(c["qV"] * (c["Tf"] - T) + c["mdH"] * rate + c["UAcoup"] * (Tc - T))


def ca_model_ss(T, c):
    """모델(n=1)의 정상상태 C_A — 해석해가 있어 로그의 T만으로 계산 가능(C_A 미측정 가정)."""
    kk = k_of(T, c["kref"], c["ER"])
    return c["qV"] * c["CAf"] / (c["qV"] + kk)


# ══════════════════════════════════════════════════════ 제어기
# 공통 규약: act(T, T_sp, ...) -> 적용할 T_c. 내부에서 rate limit·범위 클립을 건다.
# 모든 MPC는 동일한 축약(호라이즌 동안 T_c 일정 = 이동블록 1개)과 동일한 탐색기를 쓴다 —
# 계산량 때문이며, 변종 간 비교는 그 축약을 공유하므로 공정하다.

class PIController:
    """속도형 PI. T_c↑ → T↑ 이므로 e = T_sp − T 에 양의 이득."""

    def __init__(self, Kc=0.9, tauI=2.0, dt=DT):
        self.Kc, self.tauI, self.dt = Kc, tauI, dt
        self.Tc, self.e_prev = 0.0, None

    def reset(self, Tc0):
        self.Tc = float(Tc0)
        self.e_prev = None

    def act(self, T, T_sp, **_):
        # PI는 model-free이므로 앵커(r_hat)를 받지 않는다 — 시그니처만 맞춘다.
        e = T_sp - T
        de = 0.0 if self.e_prev is None else (e - self.e_prev)
        self.e_prev = e
        dTc = self.Kc * de + (self.Kc / self.tauI) * e * self.dt   # 속도형 → windup 없음
        dTc = float(np.clip(dTc, -TC_RATE, TC_RATE))
        self.Tc = float(np.clip(self.Tc + dTc, TC_LO, TC_HI))
        return self.Tc


class _MPCBase:
    """비선형 모델 예측 + 격자 탐색(2단 조밀화)."""

    def __init__(self, calib, H=12, rho=0.05, dt=DT, n_grid=13, offset_free=False,
                 dist_rho=0.85):
        self.c = dict(calib)
        self.H, self.rho, self.dt, self.n_grid = H, rho, dt, n_grid
        self.offset_free, self.dist_rho = offset_free, dist_rho
        self.Tc, self.dhat, self._T_pred = 0.0, 0.0, None

    def reset(self, Tc0):
        self.Tc = float(Tc0)
        self.dhat = 0.0
        self._T_pred = None

    def _predict(self, CA, T, Tc, calib):
        """T_c를 호라이즌 동안 일정하게 유지했을 때의 T 궤적."""
        traj = []
        for _ in range(self.H):
            CA, T = advance(CA, T, Tc, lambda x, t, u: deriv_model(x, t, u, calib), dt=self.dt)
            traj.append(T)
        return traj

    def act(self, T, T_sp, r_hat=0.0, **_):
        # 레짐 앵커 주입: 컨트롤러의 내부 모델도 그 레짐의 운전 레벨을 알아야 한다.
        # 건조 사례에서 V2Controller가 매 스텝 u_off를 주입하는 것과 같은 역할이며,
        # 이것이 없으면 MPC는 레벨이 어긋난 모델로 예측해 구조적으로 불리해진다.
        # 값은 **nominal 앵커**(로그로 적합한 것)만 쓴다 — 멤버 물리는 알지 못한다.
        # 오프셋프리 보정: 1스텝 예측오차를 **rate 단위**로 환산해 누적한다.
        # r은 dT/dt에 더해지는 항(K/min)이므로 (T − T_pred)[K]를 dt로 나눠야 단위가 맞는다.
        # ★적분형이어야 한다. 지수이동평균(d̂←ρd̂+(1−ρ)innov)은 innovation이 잔여오차에
        #  비례하므로 고정점이 r의 절반 부근에 머물러 정상상태 offset을 지우지 못한다.
        if self.offset_free and self._T_pred is not None:
            innov = (T - self._T_pred) / self.dt
            self.dhat = self.dhat + (1.0 - self.dist_rho) * innov
        calib = dict(self.c)
        calib["r"] = calib["r"] + r_hat
        if self.offset_free:
            calib["r"] = calib["r"] + self.dhat
        CA = ca_model_ss(T, calib)          # C_A 미측정 → 모델 정상상태로 추정
        lo = max(TC_LO, self.Tc - TC_RATE)
        hi = min(TC_HI, self.Tc + TC_RATE)
        best, best_J = self.Tc, np.inf
        for _ in range(2):
            for u in np.linspace(lo, hi, self.n_grid):
                traj = self._predict(CA, T, float(u), calib)
                J = sum((t - T_sp) ** 2 for t in traj) + self.rho * (u - self.Tc) ** 2
                if J < best_J:
                    best_J, best = J, float(u)
            span = (hi - lo) / (self.n_grid - 1)
            lo, hi = max(TC_LO, best - span), min(TC_HI, best + span)
        self._T_pred = self._predict(CA, T, best, calib)[0]
        self.Tc = best
        return self.Tc


class NMPC(_MPCBase):
    """내부 모델 = 비선형 gray-box(E는 fix·kref는 fit·레짐 앵커 r 반영)."""


class TubeMPC(_MPCBase):
    """오프셋프리 외란 보정을 갖춘 강건 MPC 기준선(논문의 tube MPC 대응)."""

    def __init__(self, calib, **kw):
        kw.setdefault("offset_free", True)
        super().__init__(calib, **kw)


def _linearize(CA, T, c):
    """모델을 (C_A, T) 주위에서 해석적으로 선형화 → (A, B). 상태편차 예측용."""
    kk = k_of(T, c["kref"], c["ER"])
    dkdT = kk * c["ER"] / max(T, 1.0) ** 2
    n, CAp = c["n"], max(CA, 1e-9)
    rate_CA = kk * n * CAp ** (n - 1.0)          # ∂rate/∂C_A
    rate_T = dkdT * CAp ** n                     # ∂rate/∂T
    A = np.array([[-c["qV"] - rate_CA, -rate_T],
                  [c["mdH"] * rate_CA, -c["qV"] + c["mdH"] * rate_T - c["UAcoup"]]])
    B = np.array([0.0, c["UAcoup"]])
    return A, B


class GainSchedMPC(_MPCBase):
    """현재 운전점에서 **해석적으로 선형화**한 2상태 모델로 예측(레짐 스케줄링의 연속판).
    비선형 예측을 쓰지 않는 것이 NMPC와의 차이 — 논문 패널의 gain-scheduled MPC에 대응."""

    def _predict(self, CA, T, Tc, calib):
        A, B = _linearize(CA, T, calib)
        f0 = np.array(deriv_model(CA, T, self.Tc, calib))   # 선형화 기준점의 도함수
        x = np.zeros(2)                                     # 상태 편차
        du = Tc - self.Tc
        traj = []
        for _ in range(self.H):
            x = x + self.dt * (f0 + A @ x + B * du)
            traj.append(T + x[1])
        return traj


class DataLinearMPC(_MPCBase):
    """폐루프 로그에서 동정한 **단일 전역 선형** 1차 모델(스케줄링 없음)."""

    def __init__(self, gain, tau, T_op, Tc_op, **kw):
        super().__init__(NOMINAL, **kw)
        self.gain, self.tau, self.T_op, self.Tc_op = gain, tau, T_op, Tc_op

    def _predict(self, CA, T, Tc, calib):
        del CA, calib                      # 전역 선형 모델이므로 상태·물리 파라미터를 쓰지 않는다
        T_ss = self.T_op + self.gain * (Tc - self.Tc_op)
        # tau는 **스텝** 단위다. 여기에 dt를 또 곱하면 유효 시상수가 1/dt배로 늘어난다
        # (실측: 의도 1.2 min → 12 min). 곱하지 않는 것이 옳다.
        traj, Tk = [], T
        for _ in range(self.H):
            Tk = Tk + (T_ss - Tk) / max(self.tau, 1e-6)
            traj.append(Tk)
        return traj


def make_controller(name, calib, datalin=None, H=12):
    if name == "PI":
        return PIController()
    if name == "nmpc":
        return NMPC(calib, H=H)
    if name == "gainsched":
        return GainSchedMPC(calib, H=H)
    if name == "tube":
        return TubeMPC(calib, H=H)
    if name == "datalin":
        assert datalin is not None, "datalin 제어기는 동정된 (gain, tau, T_op, Tc_op) 필요"
        return DataLinearMPC(*datalin, H=H)
    raise ValueError(name)


# ══════════════════════════════════════════════════════ 환경
class CSTREnv:
    """제어기 평가용 환경. `kind='truth'`면 진짜 플랜트(n*=1.15·레짐 촉매활성 a_g),
    `kind='model'`이면 모델 클래스 세계(n=1·멤버 E·레짐 앵커 r_g).

    에피소드 = 레짐 블록의 연결(각 블록이 drying의 LOT에 대응). 블록마다 setpoint와
    촉매활성/앵커가 바뀐다. 외란은 dT/dt에 더해지는 AR(1) + 선형 drift(286과 동형)."""

    def __init__(self, blocks, kind="truth", calib=None, anchors=None, activities=None,
                 dist_rho=0.9, dist_sigma=0.35, drift_rate=0.0, dist_type="gaussian",
                 t_df=3.0, obs_noise=0.05, seed=0, ctrl_anchors=None):
        self.blocks = list(blocks)          # [(regime_id, T_sp, n_steps), ...]
        self.kind = kind
        self.calib = dict(calib) if calib is not None else dict(NOMINAL)
        self.anchors = anchors or {}        # regime -> r_g (환경 물리에 들어가는 레벨)
        # 컨트롤러 내부 모델에 주입할 앵커 — 항상 **nominal**(로그 적합)만. 멤버 물리는 모른다.
        self.ctrl_anchors = ctrl_anchors or {}
        self.activities = activities or {}  # regime -> a_g (truth 세계)
        self.dist_rho, self.dist_sigma = dist_rho, dist_sigma
        self.drift_rate, self.dist_type, self.t_df = drift_rate, dist_type, t_df
        self.obs_noise = obs_noise
        self.rng = np.random.default_rng(seed)

    def _regime_ctx(self, g):
        if self.kind == "truth":
            return dict(a=self.activities.get(g, 1.0))
        c = dict(self.calib)
        c["r"] = self.anchors.get(g, 0.0)
        return dict(calib=c)

    def _deriv(self, ctx):
        if self.kind == "truth":
            a = ctx["a"]
            return lambda CA, T, Tc: deriv_true(CA, T, Tc, a)
        c = ctx["calib"]
        return lambda CA, T, Tc: deriv_model(CA, T, Tc, c)

    def _draw(self):
        if self.dist_type == "heavy_tail":
            return self.dist_sigma * self.rng.standard_t(self.t_df) / np.sqrt(
                self.t_df / (self.t_df - 2.0))
        return self.dist_sigma * self.rng.standard_normal()

    def steady_start(self, g, T_sp):
        """블록 시작을 그 레짐·그 setpoint의 정상상태로 둔다(제어기 간 공정한 출발점)."""
        ctx = self._regime_ctx(g)
        f = self._deriv(ctx)
        tc = tc_for_setpoint(T_sp, lambda u: _settle(f, u))
        return _settle(f, tc) + (tc,)

    def run(self, controller):
        """전 블록을 순차 실행하고 스텝별 (T, T_sp, T_c)를 반환."""
        Ts, SPs, Tcs = [], [], []
        d = 0.0
        t_global = 0
        first = True
        for g, T_sp, n in self.blocks:
            ctx = self._regime_ctx(g)
            f = self._deriv(ctx)
            if first:
                CA, T, tc0 = self.steady_start(g, T_sp)
                controller.reset(tc0)
                first = False
            for _ in range(n):
                innov = self._draw()
                d = self.dist_rho * d + innov * np.sqrt(1 - self.dist_rho ** 2)
                dist = d + self.drift_rate * t_global
                T_obs = T + self.obs_noise * self.rng.standard_normal()
                Tc = controller.act(T_obs, T_sp, r_hat=self.ctrl_anchors.get(g, 0.0))
                CA, T = advance(CA, T, Tc, lambda x, t, u: _add_rate(f(x, t, u), dist))
                Ts.append(T)
                SPs.append(T_sp)
                Tcs.append(Tc)
                t_global += 1
        return np.array(Ts), np.array(SPs), np.array(Tcs)


def _add_rate(dxdT, dist):
    return (dxdT[0], dxdT[1] + dist)


def _settle(f, Tc, iters=400):
    CA, T = 0.5, 350.0
    for _ in range(iters):
        CA, T = advance(CA, T, Tc, f, dt=0.5, n_sub=5)
    return CA, T


def kpi(Ts, SPs, Tcs, tau_T):
    dev = Ts - SPs
    ad = np.abs(dev)
    n = max(1, len(dev))
    return dict(in_band=float(np.mean(ad <= tau_T)), bias=float(np.mean(dev)),
                cvar5=float(np.mean(np.sort(ad)[-max(1, n // 20):])),
                tv=float(np.sum(np.abs(np.diff(Tcs)))),
                normTc=float(np.mean((Tcs - TC_LO) / (TC_HI - TC_LO))))
