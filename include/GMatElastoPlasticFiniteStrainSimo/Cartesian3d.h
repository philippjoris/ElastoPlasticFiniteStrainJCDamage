/**
\file
\copyright Copyright. Tom de Geus. All rights reserved.
\license This project is released under the MIT License.
*/

#ifndef GMATELASTOPLASTICFINITESTRAINSIMO_CARTESIAN3D_H
#define GMATELASTOPLASTICFINITESTRAINSIMO_CARTESIAN3D_H

#include <GMatTensor/Cartesian3d.h>

#include "config.h"
#include "version.h"

namespace GMatElastoPlasticFiniteStrainSimo {

/**
Implementation in a 3-d Cartesian coordinate frame.
*/
namespace Cartesian3d {

/**
Von Mises equivalent strain: norm of strain deviator

\f$ \sqrt{\frac{2}{3} (dev(A))_{ij} (dev(A))_{ji}} \f$

To write to allocated data use epseq().

\param A [..., 3, 3] array.
\return [...] array.
*/
template <class T>
inline auto Epseq(const T& A) -> typename GMatTensor::allocate<xt::get_rank<T>::value - 2, T>::type
{
    GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(A.dimension() >= 2);
    GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(A.shape(A.dimension() - 1) == 3);
    GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(A.shape(A.dimension() - 2) == 3);

    using return_type = typename GMatTensor::allocate<xt::get_rank<T>::value - 2, T>::type;
    return_type ret = GMatTensor::Cartesian3d::Norm_deviatoric(A);
    ret *= std::sqrt(2.0 / 3.0);
    return ret;
}

/**
Same as epseq(), but writes to externally allocated output.

\param A [..., 3, 3] array.
\param ret output [...] array
*/
template <class T, class U>
inline void epseq(const T& A, U& ret)
{
    GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(A.dimension() >= 2);
    GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(A.shape(A.dimension() - 1) == 3);
    GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(A.shape(A.dimension() - 2) == 3);
    GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(xt::has_shape(A, ret.shape()));

    GMatTensor::Cartesian3d::norm_deviatoric(A, ret);
    ret *= std::sqrt(2.0 / 3.0);
}

/**
Von Mises equivalent stress: norm of strain deviator

\f$ \sqrt{\frac{3}{2} (dev(A))_{ij} (dev(A))_{ji}} \f$

To write to allocated data use sigeq().

\param A [..., 3, 3] array.
\return [...] array.
*/
template <class T>
inline auto Sigeq(const T& A) -> typename GMatTensor::allocate<xt::get_rank<T>::value - 2, T>::type
{
    GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(A.dimension() >= 2);
    GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(A.shape(A.dimension() - 1) == 3);
    GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(A.shape(A.dimension() - 2) == 3);

    using return_type = typename GMatTensor::allocate<xt::get_rank<T>::value - 2, T>::type;
    return_type ret = GMatTensor::Cartesian3d::Norm_deviatoric(A);
    ret *= std::sqrt(1.5);
    return ret;
}

/**
Same as Sigeq(), but writes to externally allocated output.

\param A [..., 3, 3] array.
\param ret output [...] array
*/
template <class T, class U>
inline void sigeq(const T& A, U& ret)
{
    GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(A.dimension() >= 2);
    GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(A.shape(A.dimension() - 1) == 3);
    GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(A.shape(A.dimension() - 2) == 3);
    GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(xt::has_shape(A, ret.shape()));

    GMatTensor::Cartesian3d::norm_deviatoric(A, ret);
    ret *= std::sqrt(1.5);
}

// Deformation gradient tensor -> strain = 0.5 * ln(B), B = F . F^T

/**
Strain tensor from deformation gradient tensor:

\f$ \ln(\bm{B}) / 2 \f$

with

\f$ \bm{B} = \bm{F} \cdot \bm{F}^T \f$

To write to allocated data use strain().

\param A [..., 3, 3] array.
\return [...] array.
*/
template <class T>
inline auto Strain(const T& A)
{
    auto ret = GMatTensor::Cartesian3d::A2_dot_A2T(A);
    GMatTensor::Cartesian3d::logs(ret, ret);
    ret *= 0.5;
    return ret;
}

/**
Same as Strain(), but writes to externally allocated output.

\param A [..., 3, 3] array.
\param ret output [...] array
*/
template <class T, class U>
inline void strain(const T& A, U& ret)
{
    GMatTensor::Cartesian3d::A2_dot_A2T(A, ret);
    GMatTensor::Cartesian3d::logs(ret, ret);
    ret *= 0.5;
}

/**
Array of material points with a elastic constitutive response.
\tparam N Rank of the array.
*/
template <size_t N>
class Elastic : public GMatTensor::Cartesian3d::Array<N> {
protected:
    array_type::tensor<double, N> m_K; ///< Bulk modulus per item.
    array_type::tensor<double, N> m_G; ///< Shear modulus per item.
    array_type::tensor<double, N + 2> m_F; ///< Deformation gradient tensor per item.
    array_type::tensor<double, N + 2> m_Sig; ///< Cauchy stress tensor per item.
    array_type::tensor<double, N + 4> m_C; ///< Tangent per item.

    using GMatTensor::Cartesian3d::Array<N>::m_ndim;
    using GMatTensor::Cartesian3d::Array<N>::m_stride_tensor2;
    using GMatTensor::Cartesian3d::Array<N>::m_stride_tensor4;
    using GMatTensor::Cartesian3d::Array<N>::m_size;
    using GMatTensor::Cartesian3d::Array<N>::m_shape;
    using GMatTensor::Cartesian3d::Array<N>::m_shape_tensor2;
    using GMatTensor::Cartesian3d::Array<N>::m_shape_tensor4;

public:
    using GMatTensor::Cartesian3d::Array<N>::rank;

    Elastic() = default;

    /**
    Construct system.
    \param K Bulk modulus per item.
    \param G Shear modulus per item.
    */
    template <class T>
    Elastic(const T& K, const T& G)
    {
        GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(K.dimension() == N);
        GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(xt::has_shape(K, G.shape()));
        std::copy(K.shape().cbegin(), K.shape().cend(), m_shape.begin());
        this->init(m_shape);

        m_K = K;
        m_G = G;
        m_F = this->I2();
        m_Sig = xt::empty<double>(m_shape_tensor2);
        m_C = xt::empty<double>(m_shape_tensor4);
        this->refresh();
    }

    /**
    Bulk modulus per item.
    \return [shape()].
    */
    const array_type::tensor<double, N>& K() const
    {
        return m_K;
    }

    /**
    Shear modulus per item.
    \return [shape()].
    */
    const array_type::tensor<double, N>& G() const
    {
        return m_G;
    }

    /**
    Set deformation gradient tensors.
    Internally, this calls refresh() to update stress.
    \tparam T e.g. `array_type::tensor<double, N + 2>`
    \param arg Deformation gradient tensor per item [shape(), 3, 3].
    */
    template <class T>
    void set_F(const T& arg)
    {
        GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(xt::has_shape(arg, m_shape_tensor2));
        std::copy(arg.cbegin(), arg.cend(), m_F.begin());
        this->refresh();
    }

    /**
    Set deformation gradient tensors.
    Internally, this calls refresh() to update stress.
    \tparam T e.g. `array_type::tensor<double, N + 2>`
    \param arg Deformation gradient tensor per item [shape(), 3, 3].
    \param compute_tangent Compute tangent.
    */
    template <class T>
    void set_F(const T& arg, bool compute_tangent)
    {
        GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(xt::has_shape(arg, m_shape_tensor2));
        std::copy(arg.cbegin(), arg.cend(), m_F.begin());
        this->refresh(compute_tangent);
    }

    /**
    Recompute stress from deformation gradient tensor.

    From C++, this function need **never** be called: the API takes care of this.

    For Python, this function should **only** be called when you modify elements of Eps().
    For example

        mat.F[e, q, 0, 1] = value
        ...
        mat.refresh() # "F" was changed without "mat" knowing

    Instead, if you write an nd-array, the API takes care of the refresh. I.e.

        mat.F = new_F
        # no further action needed, "mat" was refreshed

    Note though that you can call this function as often as you like, you will only loose time.
    */
    void refresh(bool compute_tangent = true)
    {
        namespace GT = GMatTensor::Cartesian3d::pointer;

#pragma omp parallel
        {
            auto II = GMatTensor::Cartesian3d::II();
            auto I4d = GMatTensor::Cartesian3d::I4d();
            auto I4s = GMatTensor::Cartesian3d::I4s();
            auto I4rt = GMatTensor::Cartesian3d::I4rt();
            auto nI4rt = xt::eval(-1.0 * I4rt);

            double K;
            double G;

            std::array<double, m_stride_tensor2> Be;
            std::array<double, m_stride_tensor2> vec;
            std::array<double, m_ndim> Be_val;
            std::array<double, m_ndim> Eps_val;
            std::array<double, m_ndim> Epsd_val;
            std::array<double, m_ndim> Sig_val;

            std::array<size_t, 4> tensor4 = {m_ndim, m_ndim, m_ndim, m_ndim};
            array_type::tensor<double, 4> dTau_dlnBe = xt::empty<double>(tensor4);
            array_type::tensor<double, 4> dlnBe_dBe = xt::empty<double>(tensor4);
            array_type::tensor<double, 4> dBe_dLT = xt::empty<double>(tensor4);
            array_type::tensor<double, 4> Kmat = xt::empty<double>(tensor4);
            array_type::tensor<double, 4> Kgeo = xt::empty<double>(tensor4);

            auto F = xt::adapt(m_F.data(), {m_ndim, m_ndim});
            auto Sig = xt::adapt(m_Sig.data(), {m_ndim, m_ndim});
            auto C = xt::adapt(m_C.data(), {m_ndim, m_ndim, m_ndim, m_ndim});

#pragma omp for
            for (size_t i = 0; i < m_size; ++i) {

                K = m_K.flat(i);
                G = m_G.flat(i);

                F.reset_buffer(&m_F.flat(i * m_stride_tensor2), m_stride_tensor2);
                Sig.reset_buffer(&m_Sig.flat(i * m_stride_tensor2), m_stride_tensor2);
                C.reset_buffer(&m_C.flat(i * m_stride_tensor4), m_stride_tensor4);

                // volume change ratio
                double J = GT::Det(F.data());

                // Finger tensor
                GT::A2_dot_A2T(F.data(), &Be[0]);

                // eigenvalue decomposition of the trial "Be"
                GT::eigs(&Be[0], &vec[0], &Be_val[0]);

                // logarithmic strain "Eps := 0.5 ln(Be)" (in diagonalised form)
                for (size_t j = 0; j < 3; ++j) {
                    Eps_val[j] = 0.5 * std::log(Be_val[j]);
                }

                // decompose strain (in diagonalised form)
                double epsm = (Eps_val[0] + Eps_val[1] + Eps_val[2]) / 3.0;
                for (size_t j = 0; j < 3; ++j) {
                    Epsd_val[j] = Eps_val[j] - epsm;
                }

                // Cauchy stress (in diagonalised form)
                for (size_t j = 0; j < 3; ++j) {
                    Sig_val[j] = (3.0 * K * epsm + 2.0 * G * Epsd_val[j]) / J;
                }

                // compute Cauchy stress, in original coordinate frame
                GT::from_eigs(&vec[0], &Sig_val[0], Sig.data());

                if (!compute_tangent) {
                    return;
                }

                // 'linearisation' of the constitutive response
                // Use that "Tau := Ce : Eps = 0.5 * Ce : ln(Be)"
                xt::noalias(dTau_dlnBe) = 0.5 * K * II + G * I4d;
                dlnBe_dBe.fill(0.0);

                for (size_t m = 0; m < 3; ++m) {
                    for (size_t n = 0; n < 3; ++n) {

                        double gc = 2.0 * (Eps_val[n] - Eps_val[m]) / (Be_val[n] - Be_val[m]);

                        if (Be_val[m] == Be_val[n]) {
                            gc = 1.0 / Be_val[m];
                        }

                        for (size_t i = 0; i < 3; ++i) {
                            for (size_t j = 0; j < 3; ++j) {
                                for (size_t k = 0; k < 3; ++k) {
                                    for (size_t l = 0; l < 3; ++l) {
                                        dlnBe_dBe(i, j, k, l) += gc * vec[i * 3 + m] *
                                                                 vec[j * 3 + n] * vec[k * 3 + m] *
                                                                 vec[l * 3 + n];
                                    }
                                }
                            }
                        }
                    }
                }

                // linearization of "Be"
                // Use that "dBe = 2 * (I4s . Be) : LT" (where "LT" refers to "L_\delta^T")
                // Hence: "dBe_dLT = 2 * (I4s * Be)"
                GT::A4_dot_B2(I4s.data(), &Be[0], dBe_dLT.data());
                dBe_dLT *= 2.0;

                // material tangent stiffness
                // Kmat = dTau_dlnBe : dlnBe_dBe : dBe_dLT
                GT::A4_ddot_B4_ddot_C4(
                    dTau_dlnBe.data(), dlnBe_dBe.data(), dBe_dLT.data(), Kmat.data());

                // geometrically non-linear tangent
                // Kgeo = -I4rt . Tau
                GT::A4_dot_B2(nI4rt.data(), Sig.data(), Kgeo.data());

                // combine tangents:
                xt::noalias(C) = Kgeo + Kmat / J;
            }
        }
    }

    /**
    Strain tensor per item.
    \return [shape(), 3, 3].
    */
    const array_type::tensor<double, N + 2>& F() const
    {
        return m_F;
    }

    /**
    Strain tensor per item.
    The user is responsible for calling refresh() after modifying entries.
    \return [shape(), 3, 3].
    */
    array_type::tensor<double, N + 2>& F()
    {
        return m_F;
    }

    /**
    Stress tensor per item.
    \return [shape(), 3, 3].
    */
    const array_type::tensor<double, N + 2>& Sig() const
    {
        return m_Sig;
    }

    /**
    Tangent tensor per item.
    \return [shape(), 3, 3, 3, 3].
    */
    const array_type::tensor<double, N + 4>& C() const
    {
        return m_C;
    }
};

/**
Array of material points with a elastic constitutive response.
\tparam N Rank of the array.
*/
template <size_t N>
class LinearHardening : public GMatTensor::Cartesian3d::Array<N> {
protected:
    array_type::tensor<double, N> m_K; ///< Bulk modulus per item.
    array_type::tensor<double, N> m_G; ///< Shear modulus per item.
    array_type::tensor<double, N> m_tauy0; ///< Initial yield stress per item.
    array_type::tensor<double, N> m_H; ///< Hardening modulus per item.
    array_type::tensor<double, N> m_epsp; ///< Plastic strain per item.
    array_type::tensor<double, N> m_epsp_t; ///< Plastic strain at previous increment per item.
    array_type::tensor<double, N + 2> m_F; ///< Deformation gradient tensor per item.
    array_type::tensor<double, N + 2> m_F_t; ///< Deformation gradient tensor at prev inc per item.
    array_type::tensor<double, N + 2> m_Be; ///< Elastic Finger tensor per item.
    array_type::tensor<double, N + 2> m_Be_t; ///< Elastic Finger tensor at prev inc per item.
    array_type::tensor<double, N + 2> m_Sig; ///< Cauchy stress tensor per item.
    array_type::tensor<double, N + 4> m_C; ///< Tangent per item.

    using GMatTensor::Cartesian3d::Array<N>::m_ndim;
    using GMatTensor::Cartesian3d::Array<N>::m_stride_tensor2;
    using GMatTensor::Cartesian3d::Array<N>::m_stride_tensor4;
    using GMatTensor::Cartesian3d::Array<N>::m_size;
    using GMatTensor::Cartesian3d::Array<N>::m_shape;
    using GMatTensor::Cartesian3d::Array<N>::m_shape_tensor2;
    using GMatTensor::Cartesian3d::Array<N>::m_shape_tensor4;

public:
    using GMatTensor::Cartesian3d::Array<N>::rank;

    LinearHardening() = default;

    /**
    Construct system.
    \param K Bulk modulus per item.
    \param G Shear modulus per item.
    \param tauy0 Initial yield stress per item.
    \param H Hardening modulus per item.
    */
    template <class T>
    LinearHardening(const T& K, const T& G, const T& tauy0, const T& H)
    {
        GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(K.dimension() == N);
        GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(xt::has_shape(K, G.shape()));
        std::copy(K.shape().cbegin(), K.shape().cend(), m_shape.begin());
        this->init(m_shape);

        m_K = K;
        m_G = G;
        m_tauy0 = tauy0;
        m_H = H;
        m_epsp = xt::zeros<double>(m_shape);
        m_epsp_t = m_epsp;
        m_F = this->I2();
        m_F_t = m_F;
        m_Be = m_F;
        m_Be_t = m_F;
        m_Sig = xt::empty<double>(m_shape_tensor2);
        m_C = xt::empty<double>(m_shape_tensor4);
        this->refresh();
    }

    /**
    Bulk modulus per item.
    \return [shape()].
    */
    const array_type::tensor<double, N>& K() const
    {
        return m_K;
    }

    /**
    Shear modulus per item.
    \return [shape()].
    */
    const array_type::tensor<double, N>& G() const
    {
        return m_G;
    }

    /**
    Initial yield stress per item.
    \return [shape()].
    */
    const array_type::tensor<double, N>& tauy0() const
    {
        return m_tauy0;
    }

    /**
    Hardening modulus per item.
    \return [shape()].
    */
    const array_type::tensor<double, N>& H() const
    {
        return m_H;
    }

    /**
    Set deformation gradient tensors.
    Internally, this calls refresh() to update stress.
    \tparam T e.g. `array_type::tensor<double, N + 2>`
    \param arg Deformation gradient tensor per item [shape(), 3, 3].
    */
    template <class T>
    void set_F(const T& arg)
    {
        GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(xt::has_shape(arg, m_shape_tensor2));
        std::copy(arg.cbegin(), arg.cend(), m_F.begin());
        this->refresh();
    }

    /**
    Set deformation gradient tensors.
    Internally, this calls refresh() to update stress.
    \tparam T e.g. `array_type::tensor<double, N + 2>`
    \param arg Deformation gradient tensor per item [shape(), 3, 3].
    \param compute_tangent Compute tangent.
    */
    template <class T>
    void set_F(const T& arg, bool compute_tangent)
    {
        GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(xt::has_shape(arg, m_shape_tensor2));
        std::copy(arg.cbegin(), arg.cend(), m_F.begin());
        this->refresh(compute_tangent);
    }

    /**
    Recompute stress from deformation gradient tensor.

    From C++, this function need **never** be called: the API takes care of this.

    For Python, this function should **only** be called when you modify elements of Eps().
    For example

        mat.F[e, q, 0, 1] = value
        ...
        mat.refresh() # "F" was changed without "mat" knowing

    Instead, if you write an nd-array, the API takes care of the refresh. I.e.

        mat.F = new_F
        # no further action needed, "mat" was refreshed

    Note though that you can call this function as often as you like, you will only loose time.
    */
    void refresh(bool compute_tangent = true)
    {
        namespace GT = GMatTensor::Cartesian3d::pointer;

#pragma omp parallel
        {
            auto II = GMatTensor::Cartesian3d::II();
            auto I4d = GMatTensor::Cartesian3d::I4d();
            auto I4s = GMatTensor::Cartesian3d::I4s();
            auto I4rt = GMatTensor::Cartesian3d::I4rt();
            auto nI4rt = xt::eval(-1.0 * I4rt);

            double K;
            double G;
            double tauy0;
            double H;
            double epsp_t;

            std::array<double, m_stride_tensor2> Finv_t;
            std::array<double, m_stride_tensor2> Fdelta;
            std::array<double, m_stride_tensor2> Be_trial;
            std::array<double, m_stride_tensor2> N2;
            std::array<double, m_stride_tensor2> vec;
            std::array<double, m_ndim> Be_trial_val;
            std::array<double, m_ndim> Epse_val;
            std::array<double, m_ndim> Epsed_val;
            std::array<double, m_ndim> Taud_val;
            std::array<double, m_ndim> Sig_val;
            std::array<double, m_ndim> N_val;
            std::array<double, m_ndim> lnBe_val;

            std::array<size_t, 4> tensor4 = {m_ndim, m_ndim, m_ndim, m_ndim};
            array_type::tensor<double, 4> NN = xt::empty<double>(tensor4);
            array_type::tensor<double, 4> dTau_dlnBe = xt::empty<double>(tensor4);
            array_type::tensor<double, 4> dlnBe_dBe = xt::empty<double>(tensor4);
            array_type::tensor<double, 4> dBe_dLT = xt::empty<double>(tensor4);
            array_type::tensor<double, 4> Kmat = xt::empty<double>(tensor4);
            array_type::tensor<double, 4> Kgeo = xt::empty<double>(tensor4);

            auto F = xt::adapt(m_F.data(), {m_ndim, m_ndim});
            auto F_t = xt::adapt(m_F_t.data(), {m_ndim, m_ndim});
            auto Be = xt::adapt(m_Be.data(), {m_ndim, m_ndim});
            auto Be_t = xt::adapt(m_Be_t.data(), {m_ndim, m_ndim});
            auto Sig = xt::adapt(m_Sig.data(), {m_ndim, m_ndim});
            auto C = xt::adapt(m_C.data(), {m_ndim, m_ndim, m_ndim, m_ndim});

#pragma omp for
            for (size_t i = 0; i < m_size; ++i) {

                K = m_K.flat(i);
                G = m_G.flat(i);
                tauy0 = m_tauy0.flat(i);
                H = m_H.flat(i);
                epsp_t = m_epsp_t.flat(i);

                F.reset_buffer(&m_F.flat(i * m_stride_tensor2), m_stride_tensor2);
                F_t.reset_buffer(&m_F_t.flat(i * m_stride_tensor2), m_stride_tensor2);
                Be.reset_buffer(&m_Be.flat(i * m_stride_tensor2), m_stride_tensor2);
                Be_t.reset_buffer(&m_Be_t.flat(i * m_stride_tensor2), m_stride_tensor2);
                Sig.reset_buffer(&m_Sig.flat(i * m_stride_tensor2), m_stride_tensor2);
                C.reset_buffer(&m_C.flat(i * m_stride_tensor4), m_stride_tensor4);

                // volume change ratio
                double J = GT::Det(F.data());

                // inverse of "F_t"
                GT::Inv(F_t.data(), &Finv_t[0]);

                // incremental deformation gradient tensor
                GT::A2_dot_B2(F.data(), &Finv_t[0], &Fdelta[0]);

                // trial elastic Finger tensor (symmetric)
                // assumes "Fdelta" to result in only elastic deformation: corrected below if needed
                GT::A2_dot_B2_dot_C2T(&Fdelta[0], Be_t.data(), &Fdelta[0], Be.data());

                // copy trial elastic Finger tensor (not updated by the return map)
                std::copy(Be.begin(), Be.end(), Be_trial.begin());

                // eigenvalue decomposition of the trial "Be"
                GT::eigs(&Be_trial[0], &vec[0], &Be_trial_val[0]);

                // logarithmic strain "Eps := 0.5 ln(Be)" (in diagonalised form)
                for (size_t j = 0; j < 3; ++j) {
                    Epse_val[j] = 0.5 * std::log(Be_trial_val[j]);
                }

                // decompose strain (in diagonalised form)
                double epsem = (Epse_val[0] + Epse_val[1] + Epse_val[2]) / 3.0;
                for (size_t j = 0; j < 3; ++j) {
                    Epsed_val[j] = Epse_val[j] - epsem;
                }

                // decomposed trial (equivalent) Kirchhoff stress (in diagonalised form)
                double taum = 3.0 * K * epsem;
                for (size_t j = 0; j < 3; ++j) {
                    Taud_val[j] = 2.0 * G * Epsed_val[j];
                }
                double taueq = std::sqrt(
                    1.5 * (std::pow(Taud_val[0], 2.0) + std::pow(Taud_val[1], 2.0) +
                           std::pow(Taud_val[2], 2.0)));

                // evaluate the yield surface
                double phi = taueq - (tauy0 + H * epsp_t);

                // (direction of) plastic flow
                double dgamma = 0.0;

                // return map
                if (phi > 0) {
                    // - plastic flow
                    dgamma = phi / (3.0 * G + H);
                    // - update trial stress and elastic strain (only the deviatoric part)
                    for (size_t j = 0; j < 3; ++j) {
                        N_val[j] = 1.5 * Taud_val[j] / taueq;
                        Taud_val[j] *= (1.0 - 3.0 * G * dgamma / taueq);
                        Epsed_val[j] = Taud_val[j] / (2.0 * G);
                        lnBe_val[j] = std::exp(2.0 * (epsem + Epsed_val[j]));
                    }
                    // - update elastic Finger tensor, in original coordinate frame
                    GT::from_eigs(&vec[0], &lnBe_val[0], Be.data());
                    // - update equivalent plastic strain
                    m_epsp.flat(i) = epsp_t + dgamma;
                }

                // compute Cauchy stress, in original coordinate frame
                for (size_t j = 0; j < 3; ++j) {
                    Sig_val[j] = (taum + Taud_val[j]) / J;
                }
                GT::from_eigs(&vec[0], &Sig_val[0], Sig.data());

                if (!compute_tangent) {
                    return;
                }

                // linearisation of the constitutive response
                if (phi <= 0) {
                    // - Use that "Tau := Ce : Eps = 0.5 * Ce : ln(Be)"
                    xt::noalias(dTau_dlnBe) = 0.5 * K * II + G * I4d;
                }
                else {
                    // - Directions of plastic flow
                    GT::from_eigs(&vec[0], &N_val[0], &N2[0]);
                    GT::A2_dyadic_B2(&N2[0], &N2[0], NN.data());
                    // - Temporary constants
                    double a0;
                    double a1 = G / (H + 3.0 * G);
                    if (dgamma != 0.0) {
                        a0 = dgamma * G / taueq;
                    }
                    else {
                        a0 = 0.0;
                    }
                    // - Elasto-plastic tangent
                    xt::noalias(dTau_dlnBe) = (0.5 * (K - 2.0 / 3.0 * G) + a0 * G) * II +
                                              (1.0 - 3.0 * a0) * G * I4s + 2.0 * G * (a0 - a1) * NN;
                }

                dlnBe_dBe.fill(0.0);

                for (size_t m = 0; m < 3; ++m) {
                    for (size_t n = 0; n < 3; ++n) {

                        double gc = (std::log(Be_trial_val[n]) - std::log(Be_trial_val[m])) /
                                    (Be_trial_val[n] - Be_trial_val[m]);

                        if (Be_trial_val[m] == Be_trial_val[n]) {
                            gc = 1.0 / Be_trial_val[m];
                        }

                        for (size_t i = 0; i < 3; ++i) {
                            for (size_t j = 0; j < 3; ++j) {
                                for (size_t k = 0; k < 3; ++k) {
                                    for (size_t l = 0; l < 3; ++l) {
                                        dlnBe_dBe(i, j, k, l) += gc * vec[i * 3 + m] *
                                                                 vec[j * 3 + n] * vec[k * 3 + m] *
                                                                 vec[l * 3 + n];
                                    }
                                }
                            }
                        }
                    }
                }

                // linearization of "Be"
                // Use that "dBe = 2 * (I4s . Be) : LT" (where "LT" refers to "L_\delta^T")
                // Hence: "dBe_dLT = 2 * (I4s * Be)"
                GT::A4_dot_B2(I4s.data(), &Be_trial[0], dBe_dLT.data());
                dBe_dLT *= 2.0;

                // material tangent stiffness
                // Kmat = dTau_dlnBe : dlnBe_dBe : dBe_dLT
                GT::A4_ddot_B4_ddot_C4(
                    dTau_dlnBe.data(), dlnBe_dBe.data(), dBe_dLT.data(), Kmat.data());

                // geometrically non-linear tangent
                // Kgeo = -I4rt . Tau
                GT::A4_dot_B2(nI4rt.data(), Sig.data(), Kgeo.data());

                // combine tangents:
                xt::noalias(C) = Kgeo + Kmat / J;
            }
        }
    }

    /**
    Strain tensor per item.
    \return [shape(), 3, 3].
    */
    const array_type::tensor<double, N + 2>& F() const
    {
        return m_F;
    }

    /**
    Strain tensor per item.
    The user is responsible for calling refresh() after modifying entries.
    \return [shape(), 3, 3].
    */
    array_type::tensor<double, N + 2>& F()
    {
        return m_F;
    }

    /**
    Stress tensor per item.
    \return [shape(), 3, 3].
    */
    const array_type::tensor<double, N + 2>& Sig() const
    {
        return m_Sig;
    }

    /**
    Tangent tensor per item.
    \return [shape(), 3, 3, 3, 3].
    */
    const array_type::tensor<double, N + 4>& C() const
    {
        return m_C;
    }

    /**
    Plastic strain per item.
    \return [shape()].
    */
    const array_type::tensor<double, N>& epsp() const
    {
        return m_epsp;
    }

    /**
    Update history variables.
    */
    void increment()
    {
        std::copy(m_epsp.cbegin(), m_epsp.cend(), m_epsp_t.begin());
        std::copy(m_F.cbegin(), m_F.cend(), m_F_t.begin());
        std::copy(m_Be.cbegin(), m_Be.cend(), m_Be_t.begin());
    }
};

/**
Array of material points with a elastic constitutive response.
\tparam N Rank of the array.
*/
template <size_t N>
class LinearHardeningDamage : public GMatTensor::Cartesian3d::Array<N> {
protected:
    array_type::tensor<double, N> m_K; ///< Bulk modulus per item.
    array_type::tensor<double, N> m_G; ///< Shear modulus per item.
    array_type::tensor<double, N> m_tauy0; ///< Initial yield stress per item.
    array_type::tensor<double, N> m_H; ///< Hardening modulus per item.
    array_type::tensor<double, N> m_epsp; ///< Plastic strain per item.
    array_type::tensor<double, N> m_epsp_t; ///< Plastic strain at previous increment per item.
    array_type::tensor<double, N + 2> m_F; ///< Deformation gradient tensor per item.
    array_type::tensor<double, N + 2> m_F_t; ///< Deformation gradient tensor at prev inc per item.
    array_type::tensor<double, N + 2> m_Be; ///< Elastic Finger tensor per item.
    array_type::tensor<double, N + 2> m_Be_t; ///< Elastic Finger tensor at prev inc per item.
    array_type::tensor<double, N + 2> m_Sig; ///< Cauchy stress tensor per item.
    array_type::tensor<double, N + 4> m_C; ///< Tangent per item.

    // internal Johnson-Cook Damage variable
    array_type::tensor<double, N> m_D1; ///< JC damage parameter D1 per item.
    array_type::tensor<double, N> m_D2; ///< JC damage parameter D2 per item.
    array_type::tensor<double, N> m_D3; ///< JC damage parameter D3 per item.

    array_type::tensor<double, N> m_eps0_dot; ///< Reference plastic strain rate per item.
    array_type::tensor<double, N> m_D_damage; ///< Accumulated damage variable per item.
    array_type::tensor<double, N> m_D_damage_t; ///< Accumulated damage variable at previous increment per item.
    array_type::tensor<double, N> m_epsp_dot; ///< Equivalent plastic strain rate per item.
    array_type::tensor<double, N> m_epsp_dot_t; ///< Equivalent plastic strain rate at previous increment per item.

    array_type::tensor<bool, N> m_failed; ///< Flag indicating if element has failed.


    using GMatTensor::Cartesian3d::Array<N>::m_ndim;
    using GMatTensor::Cartesian3d::Array<N>::m_stride_tensor2;
    using GMatTensor::Cartesian3d::Array<N>::m_stride_tensor4;
    using GMatTensor::Cartesian3d::Array<N>::m_size;
    using GMatTensor::Cartesian3d::Array<N>::m_shape;
    using GMatTensor::Cartesian3d::Array<N>::m_shape_tensor2;
    using GMatTensor::Cartesian3d::Array<N>::m_shape_tensor4;

public:
    using GMatTensor::Cartesian3d::Array<N>::rank;

    LinearHardeningDamage() = default;

    /**
    Construct system.
    \param K Bulk modulus per item.
    \param G Shear modulus per item.
    \param tauy0 Initial yield stress per item.
    \param H Hardening modulus per item.
    */
    template <class T>
    LinearHardeningDamage(const T& K,
        const T& G,
        const T& tauy0,
        const T& H,
        const T& D1_in,      // Add D1 input
        const T& D2_in,      // Add D2 input
        const T& D3_in,      // Add D3 input
        const T& eps0_dot_in // Add reference strain rate input
    )
    {
        GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(K.dimension() == N);
        GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(xt::has_shape(K, G.shape()));
        GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(xt::has_shape(K, D1_in.shape()));
        GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(xt::has_shape(K, D2_in.shape()));
        GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(xt::has_shape(K, D3_in.shape()));
        GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(xt::has_shape(K, eps0_dot_in.shape()));        

        std::copy(K.shape().cbegin(), K.shape().cend(), m_shape.begin());
        this->init(m_shape);

        m_K = K;
        m_G = G;
        m_tauy0 = tauy0;
        m_H = H;

        m_D1 = D1_in;
        m_D2 = D2_in;
        m_D3 = D3_in;
        m_eps0_dot = eps0_dot_in;

        m_epsp = xt::zeros<double>(m_shape);
        m_epsp_t = m_epsp;

        m_D_damage = xt::zeros<double>(m_shape);
        m_D_damage_t = m_D_damage;
        m_epsp_dot = xt::zeros<double>(m_shape); 
        m_epsp_dot_t = m_epsp_dot;              
        m_failed = xt::zeros<bool>(m_shape);

        m_F = this->I2();
        m_F_t = m_F;
        m_Be = m_F;
        m_Be_t = m_F;
        m_Sig = xt::empty<double>(m_shape_tensor2);
        m_C = xt::empty<double>(m_shape_tensor4);
        this->refresh();
    }

    /**
    Bulk modulus per item.
    \return [shape()].
    */
    const array_type::tensor<double, N>& K() const
    {
        return m_K;
    }

    /**
    Shear modulus per item.
    \return [shape()].
    */
    const array_type::tensor<double, N>& G() const
    {
        return m_G;
    }

    /**
    Initial yield stress per item.
    \return [shape()].
    */
    const array_type::tensor<double, N>& tauy0() const
    {
        return m_tauy0;
    }

    /**
    Hardening modulus per item.
    \return [shape()].
    */
    const array_type::tensor<double, N>& H() const
    {
        return m_H;
    }

    /**
    Accumulated damage variable per item
    \return [shape()].
    */
    const array_type::tensor<double, N>& D_damage() const
    {
        return m_D_damage;
    }

    /**
    Flag indicating if element has failed.
    \return [shape()].
    */
    const array_type::tensor<bool, N>& failed() const
    {
        return m_failed;
    }

    /**
    Set deformation gradient tensors.
    Internally, this calls refresh() to update stress.
    \tparam T e.g. `array_type::tensor<double, N + 2>`
    \param arg Deformation gradient tensor per item [shape(), 3, 3].
    */
    template <class T>
    void set_F(const T& arg)
    {
        GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(xt::has_shape(arg, m_shape_tensor2));
        std::copy(arg.cbegin(), arg.cend(), m_F.begin());
        this->refresh();
    }

    /**
    Set deformation gradient tensors.
    Internally, this calls refresh() to update stress.
    \tparam T e.g. `array_type::tensor<double, N + 2>`
    \param arg Deformation gradient tensor per item [shape(), 3, 3].
    \param compute_tangent Compute tangent.
    */
    template <class T>
    void set_F(const T& arg, bool compute_tangent)
    {
        GMATELASTOPLASTICFINITESTRAINSIMO_ASSERT(xt::has_shape(arg, m_shape_tensor2));
        std::copy(arg.cbegin(), arg.cend(), m_F.begin());
        this->refresh(compute_tangent);
    }

    /**
    Recompute stress from deformation gradient tensor.

    From C++, this function need **never** be called: the API takes care of this.

    For Python, this function should **only** be called when you modify elements of Eps().
    For example

        mat.F[e, q, 0, 1] = value
        ...
        mat.refresh() # "F" was changed without "mat" knowing

    Instead, if you write an nd-array, the API takes care of the refresh. I.e.

        mat.F = new_F
        # no further action needed, "mat" was refreshed

    Note though that you can call this function as often as you like, you will only loose time.
    */
    void refresh(bool compute_tangent = true, bool element_erosion = false)
    {
        namespace GT = GMatTensor::Cartesian3d::pointer;

#pragma omp parallel
        {
            auto II = GMatTensor::Cartesian3d::II();
            auto I4d = GMatTensor::Cartesian3d::I4d();
            auto I4s = GMatTensor::Cartesian3d::I4s();
            auto I4rt = GMatTensor::Cartesian3d::I4rt();
            auto nI4rt = xt::eval(-1.0 * I4rt);

            double K;
            double G;
            double tauy0;
            double H;
            double epsp_t;

            double D1;
            double D2;
            double D3;
            double D4;
            double eps0_dot;
            double D_damage_t; 

            std::array<double, m_stride_tensor2> Finv_t;
            std::array<double, m_stride_tensor2> Fdelta;
            std::array<double, m_stride_tensor2> Be_trial;
            std::array<double, m_stride_tensor2> N2;
            std::array<double, m_stride_tensor2> vec;
            std::array<double, m_ndim> Be_trial_val;
            std::array<double, m_ndim> Epse_val;
            std::array<double, m_ndim> Epsed_val;
            std::array<double, m_ndim> Taud_val;
            std::array<double, m_ndim> Sig_val;
            std::array<double, m_ndim> N_val;
            std::array<double, m_ndim> lnBe_val;

            std::array<size_t, 4> tensor4 = {m_ndim, m_ndim, m_ndim, m_ndim};
            array_type::tensor<double, 4> NN = xt::empty<double>(tensor4);
            array_type::tensor<double, 4> dTau_dlnBe = xt::empty<double>(tensor4);
            array_type::tensor<double, 4> dlnBe_dBe = xt::empty<double>(tensor4);
            array_type::tensor<double, 4> dBe_dLT = xt::empty<double>(tensor4);
            array_type::tensor<double, 4> Kmat = xt::empty<double>(tensor4);
            array_type::tensor<double, 4> Kgeo = xt::empty<double>(tensor4);

            auto F = xt::adapt(m_F.data(), {m_ndim, m_ndim});
            auto F_t = xt::adapt(m_F_t.data(), {m_ndim, m_ndim});
            auto Be = xt::adapt(m_Be.data(), {m_ndim, m_ndim});
            auto Be_t = xt::adapt(m_Be_t.data(), {m_ndim, m_ndim});
            auto Sig = xt::adapt(m_Sig.data(), {m_ndim, m_ndim});
            auto C = xt::adapt(m_C.data(), {m_ndim, m_ndim, m_ndim, m_ndim});

#pragma omp for
            for (size_t i = 0; i < m_size; ++i) {

                double K_original = m_K.flat(i);
                double G_original = m_G.flat(i);
                tauy0 = m_tauy0.flat(i);
                H = m_H.flat(i);
                epsp_t = m_epsp_t.flat(i);

                D1 = m_D1.flat(i);
                D2 = m_D2.flat(i);
                D3 = m_D3.flat(i);
                D_damage_t = m_D_damage_t.flat(i);
                bool failed_prev = m_failed.flat(i);      
                
                double K_effective = K_original;
                double G_effective = G_original;

                F.reset_buffer(&m_F.flat(i * m_stride_tensor2), m_stride_tensor2);
                F_t.reset_buffer(&m_F_t.flat(i * m_stride_tensor2), m_stride_tensor2);
                Be.reset_buffer(&m_Be.flat(i * m_stride_tensor2), m_stride_tensor2);
                Be_t.reset_buffer(&m_Be_t.flat(i * m_stride_tensor2), m_stride_tensor2);
                Sig.reset_buffer(&m_Sig.flat(i * m_stride_tensor2), m_stride_tensor2);
                C.reset_buffer(&m_C.flat(i * m_stride_tensor4), m_stride_tensor4);
                
                if (failed_prev) {
                    Sig.fill(0.0);
                    C.fill(0.0);
                    m_D_damage.flat(i) = 1.0; // Ensure damage is 1.0
                    m_epsp_dot.flat(i) = 0.0; // Still good to set if kept as member
                    continue; // Skip to next material point
                }

                // volume change ratio
                double J = GT::Det(F.data());

                // inverse of "F_t"
                GT::Inv(F_t.data(), &Finv_t[0]);

                // incremental deformation gradient tensor
                GT::A2_dot_B2(F.data(), &Finv_t[0], &Fdelta[0]);

                // trial elastic Finger tensor (symmetric)
                // assumes "Fdelta" to result in only elastic deformation: corrected below if needed
                GT::A2_dot_B2_dot_C2T(&Fdelta[0], Be_t.data(), &Fdelta[0], Be.data());

                // copy trial elastic Finger tensor (not updated by the return map)
                std::copy(Be.begin(), Be.end(), Be_trial.begin());

                // eigenvalue decomposition of the trial "Be"
                GT::eigs(&Be_trial[0], &vec[0], &Be_trial_val[0]);

                // logarithmic strain "Eps := 0.5 ln(Be)" (in diagonalised form)
                for (size_t j = 0; j < 3; ++j) {
                    Epse_val[j] = 0.5 * std::log(Be_trial_val[j]);
                }

                // decompose strain (in diagonalised form)
                double epsem = (Epse_val[0] + Epse_val[1] + Epse_val[2]) / 3.0;
                for (size_t j = 0; j < 3; ++j) {
                    Epsed_val[j] = Epse_val[j] - epsem;
                }

                // --- Step 3: Apply Damage Degradation to Moduli (if element_erosion is true) ---
                if (element_erosion) {
                double degradation_factor = 1.0 - D_damage_t; // Use previous step's damage
                if (degradation_factor < 0.0) degradation_factor = 0.0;
                
                // Apply degradation only if there's actual damage from previous step
                if (D_damage_t > 0.0) {
                     K_effective *= degradation_factor;
                     G_effective *= degradation_factor;
                }
                }

                // --- Step 4: Calculate Trial Stresses with EFFECTIVE Moduli ---
                // decomposed trial (equivalent) Kirchhoff stress (in diagonalised form)
                double taum = 3.0 * K * epsem;
                for (size_t j = 0; j < 3; ++j) {
                    Taud_val[j] = 2.0 * G * Epsed_val[j];
                }
                double taueq = std::sqrt(
                    1.5 * (std::pow(Taud_val[0], 2.0) + std::pow(Taud_val[1], 2.0) +
                           std::pow(Taud_val[2], 2.0)));

                // --- Step 5: Evaluate Yield Surface ---
                double phi = taueq - (tauy0 + H * epsp_t);

                // (direction of) plastic flow
                double dgamma = 0.0;

                // --- Step 6: Return Map and Damage Accumulation ---
                if (phi > 0) {
                    // - plastic flow
                    dgamma = phi / (3.0 * G + H);
                    // - update trial stress and elastic strain (only the deviatoric part)
                    for (size_t j = 0; j < 3; ++j) {
                        N_val[j] = 1.5 * Taud_val[j] / taueq;
                        Taud_val[j] *= (1.0 - 3.0 * G * dgamma / taueq);
                        Epsed_val[j] = Taud_val[j] / (2.0 * G);
                        lnBe_val[j] = std::exp(2.0 * (epsem + Epsed_val[j]));
                    }
                    // - update elastic Finger tensor, in original coordinate frame
                    GT::from_eigs(&vec[0], &lnBe_val[0], Be.data());

                    // - update equivalent plastic strain
                    m_epsp.flat(i) = epsp_t + dgamma;
                    m_epsp_dot.flat(i) = dgamma / 1.0; // Pseudo-rate for output (since D4=0) 

                    // --- Johnson-Cook Damage Calculation ---
                    // 1. Calculate Stress Triaxiality 
                    double sigma_triaxiality = taum / taueq; 
                    if (taueq < 1e-12) { // Avoid division by zero or very small numbers
                        sigma_triaxiality = 0.0; // Or some other appropriate value if no deviatoric stress
                    }  
                    
                    // Calculate fracture strain using Johnson-Cook
                    double epsilon_f = (D1 + D2 * std::exp(D3 * sigma_triaxiality));

                    // Ensure epsilon_f is not negative or too small (physical constraint)
                    if (epsilon_f < 1e-6) {
                    epsilon_f = 1e-6; // Set a small positive value to avoid division by zero later
                    }
                    
                    double delta_D = dgamma / epsilon_f;
                    m_D_damage.flat(i) = D_damage_t + delta_D;

                    // Ensure damage does not exceed 1.0
                    if (m_D_damage.flat(i) > 1.0) {
                        m_D_damage.flat(i) = 1.0;
                        m_failed.flat(i) = true; // Mark as failed
                    }                                        
                }
                else {
                        // No plastic flow, no new damage accumulation this step
                        m_epsp_dot.flat(i) = 0.0;
                        // m_D_damage.flat(i) remains D_damage_t (no change in damage)
                        m_D_damage.flat(i) = D_damage_t; // Damage remains previous value
                }  

                // --- Step 7: Final Check for Failure after Damage Update ---
                if (m_failed.flat(i)) {
                    Sig.fill(0.0);
                    C.fill(0.0);
                    continue; // Skip rest of calculations for this failed element                    
                }

                // --- Step 8: Final Stress Calculation ---
                for (size_t j = 0; j < 3; ++j) {
                    Sig_val[j] = (taum + Taud_val[j]) / J;
                }
                GT::from_eigs(&vec[0], &Sig_val[0], Sig.data());
                
                // --- Tangent Calculation ---
                if (!compute_tangent) {
                    return;
                }

                // linearisation of the constitutive response
                // The elastic and elasto-plastic tangents need to be calculated with the *degraded* moduli (local_K, local_G)
                if (phi <= 0) { // If elastic or failed
                        xt::noalias(dTau_dlnBe) = 0.5 * K_effective * II + G_effective * I4d; // Use local_K, local_G
                }
                else {
                    // - Directions of plastic flow
                    GT::from_eigs(&vec[0], &N_val[0], &N2[0]);
                    GT::A2_dyadic_B2(&N2[0], &N2[0], NN.data());
                    // - Temporary constants
                    double a0;
                    double a1 = G / (H + 3.0 * G_effective);
                    if (dgamma != 0.0) {
                        a0 = dgamma * G_effective / taueq;
                    }
                    else {
                        a0 = 0.0;
                    }
                    // - Elasto-plastic tangent
                    xt::noalias(dTau_dlnBe) = (0.5 * (K_effective - 2.0 / 3.0 * G_effective) + a0 * G_effective) * II +
                                              (1.0 - 3.0 * a0) * G_effective * I4s + 2.0 * G_effective * (a0 - a1) * NN;
                }

                dlnBe_dBe.fill(0.0);

                for (size_t m = 0; m < 3; ++m) {
                    for (size_t n = 0; n < 3; ++n) {

                        double gc = (std::log(Be_trial_val[n]) - std::log(Be_trial_val[m])) /
                                    (Be_trial_val[n] - Be_trial_val[m]);

                        if (Be_trial_val[m] == Be_trial_val[n]) {
                            gc = 1.0 / Be_trial_val[m];
                        }

                        for (size_t i = 0; i < 3; ++i) {
                            for (size_t j = 0; j < 3; ++j) {
                                for (size_t k = 0; k < 3; ++k) {
                                    for (size_t l = 0; l < 3; ++l) {
                                        dlnBe_dBe(i, j, k, l) += gc * vec[i * 3 + m] *
                                                                 vec[j * 3 + n] * vec[k * 3 + m] *
                                                                 vec[l * 3 + n];
                                    }
                                }
                            }
                        }
                    }
                }

                // linearization of "Be"
                // Use that "dBe = 2 * (I4s . Be) : LT" (where "LT" refers to "L_\delta^T")
                // Hence: "dBe_dLT = 2 * (I4s * Be)"
                GT::A4_dot_B2(I4s.data(), &Be_trial[0], dBe_dLT.data());
                dBe_dLT *= 2.0;

                // material tangent stiffness
                // Kmat = dTau_dlnBe : dlnBe_dBe : dBe_dLT
                GT::A4_ddot_B4_ddot_C4(
                    dTau_dlnBe.data(), dlnBe_dBe.data(), dBe_dLT.data(), Kmat.data());

                // geometrically non-linear tangent
                // Kgeo = -I4rt . Tau
                GT::A4_dot_B2(nI4rt.data(), Sig.data(), Kgeo.data());

                // combine tangents:
                xt::noalias(C) = Kgeo + Kmat / J;

                // Apply damage degradation to the tangent explicitly, if it wasn't fully captured by K_eff/G_eff
                // This is important to ensure the tangent goes to zero as damage approaches 1.
                // This might be redundant if the previous steps correctly use the degraded moduli,
                // but it's a safety measure, especially for explicit methods.
                // Final check to ensure tangent goes to zero if damage is very high
                if (element_erosion && m_D_damage.flat(i) > 0.0) {
                    double degradation_factor_tangent = 1.0 - m_D_damage.flat(i);
                    if (degradation_factor_tangent < 0.0) degradation_factor_tangent = 0.0;
                    xt::noalias(C) *= degradation_factor_tangent;
                }
            }
        }
    }

    /**
    Strain tensor per item.
    \return [shape(), 3, 3].
    */
    const array_type::tensor<double, N + 2>& F() const
    {
        return m_F;
    }

    /**
    Strain tensor per item.
    The user is responsible for calling refresh() after modifying entries.
    \return [shape(), 3, 3].
    */
    array_type::tensor<double, N + 2>& F()
    {
        return m_F;
    }

    /**
    Stress tensor per item.
    \return [shape(), 3, 3].
    */
    const array_type::tensor<double, N + 2>& Sig() const
    {
        return m_Sig;
    }

    /**
    Tangent tensor per item.
    \return [shape(), 3, 3, 3, 3].
    */
    const array_type::tensor<double, N + 4>& C() const
    {
        return m_C;
    }

    /**
    Plastic strain per item.
    \return [shape()].
    */
    const array_type::tensor<double, N>& epsp() const
    {
        return m_epsp;
    }

    /**
    Update history variables.
    */
    void increment()
    {
        std::copy(m_epsp.cbegin(), m_epsp.cend(), m_epsp_t.begin());
        std::copy(m_F.cbegin(), m_F.cend(), m_F_t.begin());
        std::copy(m_Be.cbegin(), m_Be.cend(), m_Be_t.begin());
        std::copy(m_D_damage.cbegin(), m_D_damage.cend(), m_D_damage_t.begin());
        std::copy(m_epsp_dot.cbegin(), m_epsp_dot.cend(), m_epsp_dot_t.begin());        
    }
};

} // namespace Cartesian3d
} // namespace GMatElastoPlasticFiniteStrainSimo

#endif
