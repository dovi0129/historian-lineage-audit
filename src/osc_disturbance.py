"""잡음 구동 감쇠 공진 진동 생성기 (5.33 step 고정 주파수).

근거: 345/347이 실측한 5.33 step 고정 주파수 성분. 348 C1/C2가 보인 "위상 확산·예보 불가"와
정합하도록 **결정론 사인이 아니라 잡음 구동 공진기**를 쓴다(위상이 확산해야 예보 불가가 재현된다).

회전 2상태 형태:
    [a; b]_t = ρ_eff · R(ω) · [a; b]_{t-1} + [ε_a; ε_b],   ε ~ N(0, σ²)
    d_osc,t  = a_t
극점 ρe^{±jω}. 이 형태를 쓰는 이유는 **envelope r = √(a²+b²)를 상태에서 직접 읽을 수 있어서**다
(AR(2) 스칼라 형태로는 진폭 되먹임을 걸 수 없다).

진폭 안정화(선택·k_amp>0): ρ_eff = ρ · (1 − k_amp·(r/r_target − 1))  [clip]
  - k_amp = 0 → 순수 잡음 구동 공진기. envelope는 Rayleigh → CV → 0.523 (유한 구간에선 더 작음).
  - k_amp ↑ → 진폭이 r_target 쪽으로 당겨져 CV → 0 (limit cycle에 가까워짐).
  spec §1이 등록한 난점(가우스 구동은 CV≈0.44로 타깃 0.32보다 큼)을 다루기 위한 **한 개의**
  추가 자유도다. 격자 탐색으로 맞추되, 안 맞으면 실패로 보고한다(파라미터를 더 늘리지 않는다).

★ RNG 격리 (P-a 비트 재현의 전제): 이 생성기는 **환경의 self.rng를 절대 쓰지 않는다.**
  환경 rng를 한 번이라도 당기면 amp_scale=0 에서도 관측노이즈 이후 스트림이 어긋나 원본 비트
  재현이 깨진다 — 정확성과 무관한 이유로 P-a가 실패하게 된다. 별도 스트림을 쓴다.
"""
import numpy as np

OMEGA_DEFAULT = 2.0 * np.pi / 5.33   # 5.33 step 주기 (345/347 실측)
OSC_STREAM = 90101                   # 환경 rng와 겹치지 않는 별도 스트림 식별자


class DampedResonatorDisturbance:
    """잡음 구동 감쇠 공진기. step()이 한 스텝의 진동 성분(u 단위)을 반환."""

    def __init__(self, rho=0.94, sigma=1.0e-4, omega=OMEGA_DEFAULT,
                 k_amp=0.0, r_target=None, amp_scale=1.0, seed=0):
        self.rho = float(rho)
        self.sigma = float(sigma)
        self.omega = float(omega)
        self.k_amp = float(k_amp)
        self.amp_scale = float(amp_scale)
        # 진폭 되먹임 기준값: 미지정이면 k_amp=0일 때의 정상 envelope 평균(Rayleigh 평균)
        if r_target is None:
            sr2 = self.sigma ** 2 / max(1.0 - self.rho ** 2, 1e-12)   # E[a²]=E[b²]
            r_target = np.sqrt(np.pi * sr2 / 2.0)                      # E[r] (Rayleigh)
        self.r_target = float(r_target)
        self.seed = int(seed)
        self.reset(seed)

    # ------------------------------------------------------------------
    def reset(self, seed=None):
        """상태·전용 RNG 초기화. 환경 seed와 결정론적으로 연결하되 스트림은 분리한다."""
        if seed is not None:
            self.seed = int(seed)
        self._rng = np.random.default_rng([self.seed, OSC_STREAM])
        # 정상분포에서 시작(초기 transient 제거 — 롤아웃 초반이 체계적으로 조용해지는 것 방지)
        sr = self.sigma / np.sqrt(max(1.0 - self.rho ** 2, 1e-12))
        self.a = float(self._rng.normal(0.0, sr))
        self.b = float(self._rng.normal(0.0, sr))
        self.n = 0

    def envelope(self):
        return float(np.hypot(self.a, self.b))

    def step(self):
        """한 스텝 진행하고 진동 성분(u 단위)을 반환."""
        if self.amp_scale == 0.0:
            # 성분 자체가 꺼진 경우에도 내부 상태는 진행시켜 amp_scale 연속성(P-b)을 보존한다.
            # 환경 rng는 건드리지 않으므로 원본 비트 재현에는 영향이 없다.
            pass
        r = self.envelope()
        rho_eff = self.rho
        if self.k_amp > 0.0 and self.r_target > 0.0:
            rho_eff = self.rho * (1.0 - self.k_amp * (r / self.r_target - 1.0))
            rho_eff = float(np.clip(rho_eff, 0.0, 0.999))
        c, s = np.cos(self.omega), np.sin(self.omega)
        ea, eb = self._rng.normal(0.0, self.sigma, 2)
        a_new = rho_eff * (c * self.a - s * self.b) + ea
        b_new = rho_eff * (s * self.a + c * self.b) + eb
        self.a, self.b = float(a_new), float(b_new)
        self.n += 1
        return self.amp_scale * self.a

    # ------------------------------------------------------------------
    def simulate(self, n):
        """n 스텝 궤적(진단·캘리브레이션용). 상태를 소비하므로 사본에서 쓸 것."""
        return np.array([self.step() for _ in range(n)], float)


def theoretical_std(rho, sigma):
    """k_amp=0일 때 d_osc의 정상 표준편차."""
    return sigma / np.sqrt(max(1.0 - rho ** 2, 1e-12))
