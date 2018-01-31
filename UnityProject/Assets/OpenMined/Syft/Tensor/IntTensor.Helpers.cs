using System;
using System.Threading.Tasks;

namespace OpenMined.Syft.Tensor
{
    public partial class IntTensor
    {
        private bool SameSizeDimensionsShapeAndLocation(ref IntTensor tensor)
        {            
            if (dataOnGpu != tensor.dataOnGpu)
            {
                throw new InvalidOperationException(String.Format("Tensors must be on same device : {0} != {1}.", dataOnGpu, tensor.dataOnGpu));
            }

            if (tensor.Size == 1 && Size != 1)
            {
                // should retry with scalar version
                return true;
            }
            if (tensor.Size != 1 && Size == 1)
            {
                // should retry with scalar version
                return true;
            }
            
            // Check if both tensors have same size
            if (tensor.Size != size)
            {
                throw new InvalidOperationException(String.Format("Tensors cannot be added since they have different sizes: {0} != {1}", tensor.Size, size));    
            }
            
            // Check if both tensors have same number of dimensions
            if (tensor.Shape.Length != shape.Length)
            {
                throw new InvalidOperationException(
                    String.Format("Tensors cannot be added since they have different number of dimensions: {0} != {1}", tensor.Shape.Length, shape.Length));
            }

            // Check if both tensors have same shapes
            for (var i = 0; i < shape.Length; i++)
            {
                if (shape[i] != tensor.Shape[i])
                {
                    throw new InvalidOperationException("Tensors cannot be added since they have different shapes.");
                }
            }
            return false;
        }
        
        private void AssertDim(int dim, int len)
        {
            if (dim < 0 || dim >= len)
            {
                throw new ArgumentOutOfRangeException(nameof(dim), "Must be between 0 and shape length exclusive.");
            }
        }

        private int GetDimReduceOffset(int index, int values, int stride)
        {
            return values * stride * (index / stride) + index % stride;
        }

        private void _dimForEach(
            int interations,
            int values,
            int stride,
            Action<int[], int, int> iterator
        )
        {
            MultiThread.For(interations, (i, len) =>
            {
                var temp = new int[values];

                int offset = GetDimReduceOffset(i, values, stride);

                for (int v = 0; v < values; v++)
                {
                    temp[v] = this[offset + v * stride];
                }

                iterator(temp, i, temp.Length);
            });
        }
    }
}
