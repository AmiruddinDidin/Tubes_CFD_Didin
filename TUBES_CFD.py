import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

class Convection:
    def __init__(self, u, L, nx, dt, bc_type, bc_val, ic_start=None, ic_end=None, ic_val=None):
        self.u = u
        self.L = L
        self.nx = nx
        self.dt = dt
        self.bc_type = bc_type.lower()
        self.bc_val = bc_val

        self.dx = self.L / self.nx
        self.x = np.linspace(0, self.L, self.nx + 1)
        self.T = np.zeros(self.nx + 1)
        self.T_old = None
        self.courant = (self.u * self.dt) / self.dx

        if ic_start is not None and ic_end is not None and ic_val is not None:
            self.set_IC(ic_start, ic_end, ic_val)

        self.apply_bc(self.T)

    def stability(self):
        if self.courant <= 1:
            return f"Stable (Courant = {self.courant:.2f})"
        else:
            return f"Unstable (Courant = {self.courant:.2f})"

    def set_IC(self, x_start, x_end, suhu):
        idx_start = int(x_start / self.dx)
        idx_end = int(x_end / self.dx)
        self.T[idx_start:idx_end + 1] = suhu

    def apply_bc(self, T_array):
        if self.bc_type == "dirichlet":
            T_array[0] = self.bc_val
        elif self.bc_type == "neumann":
            T_array[0] = T_array[1] - self.bc_val * self.dx
        return T_array

    def Euler_FTFS(self):
        T_new = self.T.copy()
        for i in range(1, self.nx):
            T_new[i] = self.T[i] - self.courant * (self.T[i+1] - self.T[i])
        T_new[-1] = self.T[-1] - self.courant * (self.T[-1] - self.T[-2])
        self.apply_bc(T_new)
        self.T = T_new.copy()
        return self.T

    def Euler_FTBS(self):
        T_new = self.T.copy()
        for i in range(1, self.nx+1):
            T_new[i] = self.T[i] - self.courant * (self.T[i] - self.T[i-1])
        self.apply_bc(T_new)
        self.T = T_new.copy()
        return self.T

    def Lax_Method(self):
        T_new = self.T.copy()
        for i in range(1, self.nx):
            T_new[i] = 0.5 * (self.T[i+1] + self.T[i-1]) - self.courant*0.5*(self.T[i+1] - self.T[i-1])
        T_new[-1] = self.T[-1] - self.courant * (self.T[-1] - self.T[-2])
        self.apply_bc(T_new)
        self.T = T_new.copy()
        return self.T

    def Midpoint_Leapfrog(self):
        T_new = self.T.copy()
        if self.T_old is None:
            self.T_old = self.T.copy()
            for i in range(1, self.nx + 1):
                T_new[i] = self.T[i] - self.courant * (self.T[i] - self.T[i-1])
            self.apply_bc(T_new)
            self.T = T_new.copy()
            return self.T

        for i in range(1, self.nx):
            T_new[i] = self.T_old[i] - self.courant * (self.T[i+1] - self.T[i-1])
        T_new[-1] = self.T[-1] - self.courant * (self.T[-1] - self.T[-2])
        self.apply_bc(T_new)
        self.T_old = self.T.copy()
        self.T = T_new.copy()
        return self.T

    def Lax_Wendroff(self):
        T_new = self.T.copy()
        for i in range(1, self.nx):
            T_new[i] = self.T[i] - self.courant * (self.T[i+1] - self.T[i-1]) / 2 + \
             (self.courant**2 / 2) * (self.T[i+1] - 2 * self.T[i] + self.T[i-1])
        T_new[-1] = self.T[-1] - self.courant * (self.T[-1] - self.T[-2])
        self.apply_bc(T_new)
        self.T = T_new.copy()
        return self.T

st.set_page_config(page_title="CFD Simulator", layout="wide")
st.title("1-Dimensional Convection Simulation")
st.markdown(f"Riflo Rafael Hutajulu - 13623015  \n Didin Amiruddin - 13623037")

st.sidebar.header("Parameter Fisik & Numerik")
u = st.sidebar.number_input("Kecepatan (u)", value=1.0, step=0.1)
L = st.sidebar.number_input("Panjang Domain (L)", value=1.0, step=0.1)
nx = st.sidebar.slider("Jumlah Grid (nx)", 10, 200, 50)
dt = st.sidebar.number_input("Waktu Step (dt)", value=0.01, format="%.4f")
time_iteration = st.sidebar.slider("Jumlah Iterasi", 10, 500, 50)

st.sidebar.markdown("---")
st.sidebar.header("Kondisi Batas & Awal")
bc_type = st.sidebar.selectbox("Tipe Batas (BC)", ["dirichlet", "neumann"])
bc_val = st.sidebar.number_input("Nilai Batas (Inlet)", value=20.0)

pakai_ic = st.sidebar.checkbox("Gunakan Initial Condition (IC)?", value=True)
if pakai_ic:
    ic_start = st.sidebar.slider("Posisi Awal IC", 0.0, float(L), 0.2)
    ic_end = st.sidebar.slider("Posisi Akhir IC", 0.0, float(L), 0.4)
    ic_val = st.sidebar.number_input("Suhu IC", value=100.0)
else:
    ic_start, ic_end, ic_val = None, None, None


if st.sidebar.button("Jalankan Simulasi", type="primary"):
    
    # Inisialisasi Objek
    Tes_FTFS = Convection(u, L, nx, dt, bc_type, bc_val, ic_start, ic_end, ic_val)
    Tes_FTBS = Convection(u, L, nx, dt, bc_type, bc_val, ic_start, ic_end, ic_val)
    Tes_LaxM = Convection(u, L, nx, dt, bc_type, bc_val, ic_start, ic_end, ic_val)
    Tes_Midpoint = Convection(u, L, nx, dt, bc_type, bc_val, ic_start, ic_end, ic_val)
    Tes_LaxW = Convection(u, L, nx, dt, bc_type, bc_val, ic_start, ic_end, ic_val)

    status = Tes_FTBS.stability()
    if "Unstable" in status:
        st.error(f"🚨 Peringatan: {status}. Turunkan dt atau naikkan nx!")
    else:
        st.success(f"✅ Sistem {status}.")

    err_FTFS, err_FTBS, err_LaxM, err_Midpoint, err_LaxW = [], [], [], [], []

    progress_bar = st.progress(0)

    for step in range(time_iteration):
        T_lama_FTFS = Tes_FTFS.T.copy()
        T_lama_FTBS = Tes_FTBS.T.copy()
        T_lama_LaxM = Tes_LaxM.T.copy()
        T_lama_Midpoint = Tes_Midpoint.T.copy()
        T_lama_LaxW = Tes_LaxW.T.copy()

        R_FTFS = Tes_FTFS.Euler_FTFS()
        R_FTBS = Tes_FTBS.Euler_FTBS()
        R_LaxM = Tes_LaxM.Lax_Method()
        R_Midpoint = Tes_Midpoint.Midpoint_Leapfrog()
        R_LaxW = Tes_LaxW.Lax_Wendroff()

        err_FTFS.append(np.max(np.abs(R_FTFS - T_lama_FTFS)))
        err_FTBS.append(np.max(np.abs(R_FTBS - T_lama_FTBS)))
        err_LaxM.append(np.max(np.abs(R_LaxM - T_lama_LaxM)))
        err_Midpoint.append(np.max(np.abs(R_Midpoint - T_lama_Midpoint)))
        err_LaxW.append(np.max(np.abs(R_LaxW - T_lama_LaxW)))

        progress_bar.progress((step + 1) / time_iteration)

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        x_grid = Tes_FTFS.x
        
        # ax1.plot(x_grid, R_FTFS, label="FTFS", linestyle=':', alpha=0.3)
        ax1.plot(x_grid, R_FTBS, label="FTBS (Upwind)", linewidth=2)
        ax1.plot(x_grid, R_LaxM, label="Lax Method", linewidth=2)
        ax1.plot(x_grid, R_Midpoint, label="Midpoint Leapfrog", linewidth=2)
        ax1.plot(x_grid, R_LaxW, label="Lax-Wendroff", linewidth=2)

        ax1.set_title(f"Profil Suhu pada t = {time_iteration * dt:.3f} s")
        ax1.set_xlabel("Posisi x (meter)")
        ax1.set_ylabel("Suhu")
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        eps = 1e-12 # Penyelamat log(0)
        iter_range = range(1, time_iteration + 1)

        ax2.semilogy(iter_range, np.array(err_FTFS) + eps, label="FTFS", linestyle=':', alpha=0.5)
        ax2.semilogy(iter_range, np.array(err_FTBS) + eps, label="FTBS (Upwind)", linewidth=2)
        ax2.semilogy(iter_range, np.array(err_LaxM) + eps, label="Lax Method", linewidth=2)
        ax2.semilogy(iter_range, np.array(err_Midpoint) + eps, label="Midpoint Leapfrog", linewidth=2)
        ax2.semilogy(iter_range, np.array(err_LaxW) + eps, label="Lax-Wendroff", linewidth=2)

        ax2.set_title("Grafik Konvergensi (Residual Error)")
        ax2.set_xlabel("Iterasi Waktu")
        ax2.set_ylabel("Max Absolute Error (Log Scale)")
        ax2.grid(True, which="both", linestyle='--', alpha=0.6)
        ax2.legend()
        st.pyplot(fig2)