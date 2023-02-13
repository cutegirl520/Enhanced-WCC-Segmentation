// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_VOLUME_PATCH_H
#define EIGEN_CXX11_TENSOR_TENSOR_VOLUME_PATCH_H

namespace Eigen {

/** \class TensorVolumePatch
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Patch extraction specialized for processing of volumetric data.
  * This assumes that the input has a least 4 dimensions ordered as follows:
  *  - channels
  *  - planes
  *  - rows
  *  - columns
  *  - (optional) additional dimensions such as time or batch size.
  * Calling the volume patch code with patch_planes, patch_rows, and patch_cols
  * is equivalent to calling the regular patch extraction code with parameters
  * d, patch_planes, patch_rows, patch_cols, and 1 for all the additional
  * dimensions.
  */
namespace internal {
template<DenseIndex Planes, DenseIndex Rows, DenseIndex Cols, typename XprType>
struct traits<TensorVolumePatchOp<Planes, Rows, Cols, XprType> > : public traits<XprType>
{
  typedef typename internal::remove_const<typename XprType::Scalar>::type Scalar;
  typedef traits<XprType> XprTraits;
  typedef typename XprTraits::StorageKind StorageKind;
  typedef typename XprTraits::Index Index;
  typedef typename XprType::Nested Nested;
  typedef typename remove_reference<Nested>::type _Nested;
  static const int NumDimensions = XprTraits::NumDimensions + 1;
  static const int Layout = XprTraits::Layout;
};

template<DenseIndex Planes, DenseIndex Rows, DenseIndex Cols, typename XprType>
struct eval<TensorVolumePatchOp<Planes, Rows, Cols, XprType>, Eigen::Dense>
{
  typedef const TensorVolumePatchOp<Planes, Rows, Cols, XprType>& type;
};

template<DenseIndex Planes, DenseIndex Rows, DenseIndex Cols, typename XprType>
struct nested<TensorVolumePatchOp<Planes, Rows, Cols, XprType>, 1, typename eval<TensorVolumePatchOp<Planes, Rows, Cols, XprType> >::type>
{
  typedef TensorVolumePatchOp<Planes, Rows, Cols, XprType> type;
};

}  // end namespace internal

template<DenseIndex Planes, DenseIndex Rows, DenseIndex Cols, typename XprType>
class TensorVolumePatchOp : public TensorBase<TensorVolumePatchOp<Planes, Rows, Cols, XprType>, ReadOnlyAccessors>
{
  public:
  typedef typename Eigen::internal::traits<TensorVolumePatchOp>::Scalar Scalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename Eigen::internal::nested<TensorVolumePatchOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorVolumePatchOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorVolumePatchOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorVolumePatchOp(const XprType& expr, DenseIndex patch_planes, DenseIndex patch_rows, DenseIndex patch_cols,
                                                            DenseIndex plane_strides, DenseIndex row_strides, DenseIndex col_strides,
                                                            DenseIndex in_plane_strides, DenseIndex in_row_strides, DenseIndex in_col_strides,
                                                            DenseIndex plane_inflate_strides, DenseIndex row_inflate_strides, DenseIndex col_inflate_strides,
                                                            PaddingType padding_type, Scalar padding_value)
      : m_xpr(expr), m_patch_planes(patch_planes), m_patch_rows(patch_rows), m_patch_cols(patch_cols),
        m_plane_strides(plane_strides), m_row_strides(row_strides), m_col_strides(col_strides),
        m_in_plane_strides(in_plane_strides), m_in_row_strides(in_row_strides), m_in_col_strides(in_col_strides),
        m_plane_inflate_strides(plane_inflate_strides), m_row_inflate_strides(row_inflate_strides), m_col_inflate_strides(col_inflate_strides),
        m_padding_explicit(false), m_padding_top_z(0), m_padding_bottom_z(0), m_padding_top(0), m_padding_bottom(0), m_padding_left(0), m_padding_right(0),
        m_padding_type(padding_type), m_padding_value(paddin