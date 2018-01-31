using UnityEngine;
using System;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Linq;

namespace OpenMined.Syft.Tensor
{
    public partial class IntTensor
    {
        internal IntTensor emptyTensorCopy(bool hook_graph = false, IntTensor result = null)
        {

            if (hook_graph)
            {
                result = HookGraph(ref result, "emptyTensorCopy_Hooked", false);
                result.Zero_();
                return result;
            }
            else
            {
                
                result = factory.Create(
                    _shape: this.shape,
                    _data: data,
                    _dataBuffer: dataBuffer,
                    _shapeBuffer: shapeBuffer,
                    _shader: shader,
                    _copyData: true,
                    _dataOnGpu: dataOnGpu,
                    _autograd: autograd,
                    _keepgrads: keepgrads,
                    _creation_op: "emptyTensorCopy");
            
                result.Zero_();

                return result;
            }
            
        }
        public IntTensor Add(int value, bool inline = false)
        {
            if (dataOnGpu)
            {
                throw new NotImplementedException();
            }

            IntTensor result = factory.Create(this.shape);
            result.Data = data.AsParallel().Select(x => x + value).ToArray();

            return result;
        }

        public IntTensor Add(IntTensor x, bool inline = false)
        {

            IntTensor result = factory.Create(this.shape);

            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (inline) { AddElemGPU_(x); return this; }
                else { return AddElemGPU(x, result); }
            }
            else
            {
                // run Addition on the CPU
                result.Data = data.AsParallel().Zip(x.Data.AsParallel(), (a, b) => a + b).ToArray();

                return result;
            }

        }

        public IntTensor Contiguous(IntTensor result = null)
        {

            if (DataOnGpu)
                throw new NotSupportedException();
         
            result = HookGraph(ref result, creation_op:"contiguous", inline:false, resultShape:shape);

            int[] dim_indices = new int[strides.Length];
            
            for (int i = 0; i < result.Data.Length; i++)
            {    
                result.DataIndex2DimIndices(i, ref dim_indices);
                result.data[i] = this.data[this.DimIndices2DataIndex(ref dim_indices)];
            }   
            
            return result;
        }

        public IntTensor Expand(int[] sizes) {
            if (sizes.Length == Shape.Length) {
                return ExpandFixedDimensions(sizes);
            } else if (sizes.Length > Shape.Length) {
                return expandNewDimensions(sizes);
            } else {
                throw new InvalidOperationException(String.Format("Number of sizes provided must be greater than or equal to the number of dimensions in tensor"));
            }
        }

        private IntTensor ExpandFixedDimensions(int[] sizes, IntTensor result = null)
        {

            // TODO: make more complicated version which does not copy data
            result = HookGraph(ref result, "expand", inline:false, resultShape:shape);
            result.Add(this, inline: true);
            
            for (int i = 0; i < shape.Length; i++) {
                if (sizes[i] != -1 && sizes[i] != shape[i]) {
                    if (shape[i] == 1 || strides[i] == 0) {
                        result.strides[i] = 0;
                        result.shape[i] = sizes[i];
                    } else {
                        throw new InvalidOperationException (String.Format ("Cannot expand dimension {0}, not a singleton ({1})", i, shape[i]));
                    }
                }
            }

            return result;
        }

        private IntTensor expandNewDimensions(int[] sizes) {
            IntTensor result = factory.Create(_data: data, _shape: shape, _shader: shader, _copyData: false);

            int diffLength = sizes.Length - shape.Length;
            
            // sets new strides to zero on initialization
            int[] newStrides = new int[sizes.Length];
            int[] newShape = new int[sizes.Length];

            for (int i = 0; i < diffLength; i++) {
                // sets new shape
                if (sizes[i] != -1) {
                    newShape[i] = sizes[i];
                } else {
                    throw new InvalidOperationException (String.Format ("Cannot set new dimension {0} to -1", i));
                }
            }
            
            for (int i = diffLength; i < sizes.Length; i++) {
                var oldIndex = i - diffLength;
                
                // fill in old strides/shape
                newStrides[i] = strides[oldIndex];
                newShape[i] = shape[oldIndex];
                
                // modify any old strides/shapes
                if (sizes[i] != -1 && sizes[i] != shape[oldIndex]) {
                    if (shape[oldIndex] == 1 || strides[oldIndex] == 0) {
                        newStrides[i] = 0;
                        newShape[i] = sizes[i];
                    } else {
                        throw new InvalidOperationException (String.Format ("Cannot expand dimension {0}, not a singleton ({1})", i, shape[i]));
                    }
                }
            }

            result.shape = newShape;
            result.strides = newStrides;
            
            return result;
        }


        // regular index add expects a single list as a tensor - and you select which dimension that list is
        // used to index into in a different parameter. In this method, the shape of indices itself is instead used
        // to index into the tensor - ShapedIndexSelect has a similar relationship to IndexSelect as this method has
        // to IndexAdd
        public IntTensor ShapedIndexAdd(IntTensor indices, IntTensor x, bool inline = false, IntTensor result = null)
        {
            if(indices.Shape.Length != shape.Length-1)
                throw new Exception("Indices must have exactly one dimension less than tensor");
                
            for (int i = 0; i < shape.Length-1; i++)
            {
                if (shape[i] != indices.Shape[i])
                {
                    throw new Exception(
                        "If you index select with -1, indices shape must match tensor shape for all dims except the last");
                }
            }

            int[] flat_left = this.Shape;
            
            result = HookGraph(ref result, "shaped_index_add_" + indices.Id + "_" + x.id, inline:inline, resultShape:this.Shape, indices:new IntTensor[1]{indices});
 
            /*for (int i = 0; i < result.Size; i++)
            {
                result.data[i] = this.Data[i * flat_left[1] + indices.Data[i]];
            }*/
            
            for (int i = 0; i < indices.Size; i++)
            {
                result.Data[i * flat_left[1] + indices.Data[i]] += x.Data[i];
            }

            return result;
        }

        public IntTensor IndexAdd(IntTensor indices, int dim, IntTensor x, IntTensor result = null, bool inline = false)
        {
            if (dim == -1)
            {
                return ShapedIndexAdd(indices, x, inline:inline);
            }
            
            if (DataOnGpu)
            {
                throw new NotImplementedException();
            }
            
            if (indices.Shape.Length != 1)
            {
                throw new NotImplementedException("Indices must be a list");
            }

            /*if (indices.Shape[dim] != x.Shape[dim])
            {
                throw new IndexOutOfRangeException("Indices and Input Sum must have same number of rows");
            }*/

            int[] original_shape = new int[shape.Length];
            for (int i = 0; i < shape.Length; i++) original_shape[i] = shape[i];
            
            int[] temp_shape = new int[] {1, shape[dim], 1};

            for (int i = 0; i < dim; i++)
            {
                temp_shape[0] *= shape[i];
            }
    
            for (int i = dim+1; i < shape.Length; i++)
            {
                temp_shape[2] *= shape[i];
            }
                
            var self_3d = this.View(temp_shape,inline:inline);
            var x_3d = x.View(new int[] {temp_shape[0], indices.Shape[0], temp_shape[2]});
            
            // TODO: Hook Autograd should support this
            result = HookGraph(ref result, "index_add_dim:" + dim + "_" + indices.Id + "_" + x.Id, inline, resultShape:temp_shape);

            if (!inline)
            {
                result.Zero_();
                result.Add(this, inline: true);
            }

            int[] temp_index = new int[] {0, 0, 0};
            
            for (int i = 0; i < self_3d.shape[0]; i++)
            {
                temp_index[0] = i;

                for (int j = 0; j < self_3d.shape[2]; j++)
                {
                    temp_index[2] = j;
                
                    for (var k = 0; k < indices.Shape[0]; k++)
                    {
                        temp_index[1] = k;
                        int x_dataindex = x_3d.DimIndices2DataIndex(ref temp_index);
                        
                        temp_index[1] = indices.Data[k];
                        result.Data[result.DimIndices2DataIndex(ref temp_index)] += x_3d.Data[x_dataindex];

                    }
                        
                }
                    
            }

            return result.View(original_shape, inline:inline);
        }

        public IntTensor IndexSelect(List<int> indices, int dim, IntTensor result = null)
        {
            IntTensor i = factory.ctrl.intTensorFactory.Create(_shape: new int[] {indices.Count}, _data: indices.ToArray());
            IntTensor subset = IndexSelect(i, dim, result);
            factory.ctrl.intTensorFactory.Delete(i.Id);
            return subset;
        }

        
        // this probably isn't the best/right name for this function
        // but basically the normal indexselect requies you to pass in a list of indices
        // that is exactly one dimension (a list) and a separate parameter that selects
        // which dim the list of indices should be applied to. In this one, however, indices
        // is has to same shape as the tensor itx indexing into except for the last dimension, which
        // indices must not have as a dimension (that's the one being indexed).
        public IntTensor ShapedIndexSelect(IntTensor indices, IntTensor result = null)
        {
            if(indices.Shape.Length != shape.Length-1)
                throw new Exception("Indices must have exactly one dimension less than tensor");
                
            for (int i = 0; i < shape.Length-1; i++)
            {
                if (shape[i] != indices.Shape[i])
                {
                    throw new Exception(
                        "If you index select with -1, indices shape must match tensor shape for all dims except the last");
                }
            }
                
            int[] flat_left = new int[2];
            flat_left[1] = shape[shape.Length - 1];
            flat_left[0] = 1;
            for (int i = 0; i < shape.Length - 1; i++) flat_left[i] *= shape[i];
            
            int[] slice_off_right = new int[shape.Length - 1];
            for (int i = 0; i < slice_off_right.Length; i++) slice_off_right[i] = shape[i];
            
                
            result = HookGraph(ref result, "shaped_index_select_" + indices.Id, inline:false, resultShape:slice_off_right, indices:new IntTensor[1]{indices});

            
            for (int i = 0; i < result.Size; i++)
            {
                result.data[i] = this.Data[i * flat_left[1] + indices.Data[i]];
            }

            return result;
        }
        
        
        public IntTensor IndexSelect(IntTensor indices, int dim, IntTensor result = null)
        {

            if (dim == -1)
            {
                return ShapedIndexSelect(indices);
            }
            
            if (DataOnGpu)
            {
                throw new NotImplementedException();
            }

            if (indices.Shape.Length != 1)
            {
                throw new NotImplementedException("Indices must be a list");
            }
            
            int[] temp_shape = new int[] {1, shape[dim], 1};

            for (int i = 0; i < dim; i++)
            {
                temp_shape[0] *= shape[i];
            }
    
            for (int i = dim+1; i < shape.Length; i++)
            {
                temp_shape[2] *= shape[i];
            }
                
            var self_3d = this.View(temp_shape);

            int[] result_3d_shape = new int[] {temp_shape[0], indices.Shape[0], temp_shape[2]};

            result = HookGraph(ref result, "index_select_" + dim + "_" + indices.Id, inline:false, resultShape:result_3d_shape, indices:new IntTensor[1]{indices});
            
            int[] temp_index = new int[] {0, 0, 0};
        
            for (int i = 0; i < self_3d.shape[0]; i++)
            {
                temp_index[0] = i;

                for (int j = 0; j < self_3d.shape[2]; j++)
                {
                    temp_index[2] = j;
                
                    for (var k = 0; k < indices.Shape[0]; k++)
                    {
                        temp_index[1] = indices.Data[k];
                        int result_data_index = self_3d.DimIndices2DataIndex(ref temp_index);

                        temp_index[1] = k;
                        result.Data[result.DimIndices2DataIndex(ref temp_index)] = self_3d.Data[result_data_index];
                    }       
                }                    
            }

            int[] result_dim = new int[shape.Length];
            for (int i = 0; i < shape.Length; i++)
            {
                if (i != dim)
                {
                    result_dim[i] = shape[i];
                }
                else
                {
                    result_dim[i] = indices.Shape[0];
                }
            }
            
            return result.View(result_dim);
        }
        

        public IntTensor Reciprocal(bool inline = false)
        {
            IntTensor result = factory.Create(this.shape);
            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (inline) { ReciprocalGPU_(); return this; }
                else { return ReciprocalGPU(result); }
            }
            if (inline)
            {
                this.Data = data.AsParallel().Select(x => (int)(1 / x)).ToArray();
                return this;
            }
            result.Data = data.AsParallel().Select(x => (int)(1 / x)).ToArray();
            return result;
        }


        internal IntTensor[] MakeSplits(int[] splitSections, int dim = 0)
        {
            int numSplits = splitSections.Length;
            var splits = new IntTensor[numSplits];
            int offset = 0;

            //Gather subset of elements corresponding to each split 
            for(int i = 0; i < numSplits; i++)
            {
                int[] splitShape = (int[]) Shape.Clone();
                int splitSize = splitSections[i];
                splitShape[dim] = splitSize;
                splits[i] = this.IndexSelect(new List<int>(Enumerable.Range(offset, splitSize)), dim);

                offset += splitSize;
            }
            return splits;
        }
        
        public IntTensor[] Split(int splitSize, int dim = 0)
        {
            if (!IsContiguous())
            {
                throw new InvalidOperationException ("Tensor must be contiguous, call Contiguous() to convert");
            }

            int len = shape.Length;

            AssertDim(dim, len);

            int numSplits = (shape[dim] + splitSize - 1)/splitSize;
            int lastSplitSize = splitSize - (splitSize*numSplits - shape[dim]);
            var splitSections = new int[numSplits];

            for(int i = 0; i < numSplits; i++){
                splitSections[i] = (i < (numSplits - 1)) ? splitSize : lastSplitSize;
            }

            return MakeSplits(splitSections, dim);
        }

        public IntTensor[] Split(int[] splitSections, int dim = 0)
        {

            if (!IsContiguous())
            {
                throw new InvalidOperationException ("Tensor must be contiguous, call Contiguous() to convert");
            }

            int len = shape.Length;

            AssertDim(dim, len);

            int numSplits = splitSections.Length;
            int sumSplitSizes = 0;
        
            for (int i = 0; i < numSplits; i++)
            {
                sumSplitSizes += splitSections[i];
            }

            if (sumSplitSizes != shape[dim])
            {
                throw new InvalidOperationException 
                (String.Format("Sum of split sizes {0} != size {1} of dim {2}", 
                    sumSplitSizes,  shape[dim], dim));
            }

            return MakeSplits(splitSections, dim);
        }
        
        public IntTensor Sin(bool inline = false)
        {
            IntTensor result = factory.ctrl.intTensorFactory.Create(shape);
            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (inline) { throw new NotImplementedException(); }
                else { return SinGPU(result); }
            }
            result.Data = data.AsParallel().Select(x => (int)Math.Sin((int)x)).ToArray();
            return result;
        }

        public IntTensor Sum(int dim = -1, bool keepdim = false)
        {
            if (!IsContiguous())
            {
                throw new InvalidOperationException("Tensor must be contiguous, call Contiguous() to convert");
            }

            // TODO: Implement GPU op. with GPU tests.

            return Reduce(dim, keepdim, (acc, val, index, arr) => acc + val, (val, len) => val, creation_op:"sum_"+dim+"_"+keepdim);

        }    

        public IntTensor Cos(bool inline = false)
        {
            IntTensor result = factory.ctrl.intTensorFactory.Create(shape);
            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (inline) { throw new NotImplementedException(); }
                else { return CosGPU(result); }
            }
            result.Data = data.AsParallel().Select(x => (int)Math.Cos((int)x)).ToArray();
            return result;
        }
        
        public IntTensor Eq(IntTensor other, bool inline = false)
        {
            // Run argument checks on CPU
            // Assuming broadcasting is not supported
            if (!this.shape.SequenceEqual(other.shape))
                throw new ArgumentException("Tensor dimensions must match");

            if (other == null)
                throw new ArgumentNullException();

            if (dataOnGpu) {
                throw new NotImplementedException();
            }
            else {
                if (inline) {
                    this.Data = data.AsParallel().Zip(other.Data.AsParallel(),
                                                        (a, b) => a == b ? 1 : 0).ToArray();
                    return this;
                }
                else {
                    IntTensor result = factory.Create(this.shape);
                    result.Data = data.AsParallel().Zip( other.Data.AsParallel(),
                                                        (a, b) => a == b ? 1 : 0 ).ToArray();
                    return result;
                }
            }
        }


        public IntTensor View(int[] new_shape, bool inline = true)
        {
            if (!IsContiguous())
            {
                throw new InvalidOperationException("Tensor must be contiguous, call Contiguous() to convert");
            }
            // suppport for -1 parameter value in new_shape
            var index = Array.IndexOf(new_shape, -1);
            if(index != -1)
            {
                int tempSize = 1;
                foreach(var s in new_shape)
                {
                    if (s != -1)
                        tempSize *= s;
                }
                new_shape[index] = size / tempSize;
            }

            if (inline == true)
            {

                this.Shape = new_shape;

                if (dataOnGpu)
                {
                    shapeBuffer.Release();
                    shapeBuffer = new ComputeBuffer(shape.Length, sizeof(int));
                    shapeBuffer.SetData(shape);

                }

                setStridesAndCheckShape();

                return this;

            }
            else
            {
                IntTensor result = factory.Create(new_shape);
                result.Add(this, inline: true);
                return result;
            }

        }

        public IntTensor Abs(bool inline = false)
        {
            IntTensor result = factory.Create(this.shape);

            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (inline) { AbsGPU_(); return this; }
                else { return AbsGPU(result); }
            }

            if (inline)
            {
                this.Data = data.AsParallel().Select(x => Math.Abs(x)).ToArray();
                return this;
            }
            else
            {
                result.Data = data.AsParallel().Select(x => Math.Abs(x)).ToArray();
                return result;
            }
        }

        public IntTensor Lt(IntTensor other, bool inline = false)
        {
            // Run argument checks on CPU anyway just to make sure
            if (!this.shape.SequenceEqual(other.shape))
                throw new ArgumentException("Tensor dimensions must match");

            if (other == null)
                throw new ArgumentNullException();

            if (dataOnGpu)
            {
                throw new NotImplementedException();
            }
            else
            {
                if (inline)
                {
                    this.Data = data.AsParallel().Zip(other.Data.AsParallel(),
                                                        (a, b) => a < b ? 1 : 0).ToArray();
                    return this;
                }
                else
                {
                    IntTensor result = factory.Create(this.shape);
                    result.Data = data.AsParallel().Zip(other.Data.AsParallel(),
                                                        (a, b) => a < b ? 1 : 0).ToArray();
                    return result;
                }
            }
        }

        public IntTensor Sign(bool inline = false)
        {
            IntTensor result = factory.Create(this.shape);
            if (dataOnGpu)
            {
                throw new NotImplementedException();
            }
            if (!inline)
            {
                result.Data = data.AsParallel().Select(x => (int)Math.Abs(x) / x).ToArray();
            }
            return result;
        }

        public IntTensor Sqrt(bool inline = false)
        {

            if (dataOnGpu)
            {
                return this;
            }

            IntTensor result = factory.Create(this.shape);
            result.Data = data.AsParallel().Select(x => (int)Math.Sqrt(x)).ToArray();

            return result;
        }

        public IntTensor Neg(bool inline = false, IntTensor result = null)
        {
            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (inline) { NegGPU_(); return this; }
                else { result = factory.Create(this.shape); return NegGPU(result); }
            }
            result = this;
            if (!inline) result = factory.Create(this.shape);
            result.Data = data.AsParallel().Select(x => -x).ToArray();
            return result;
        }

        public IntTensor Transpose(bool inline = false)
        {
            if (shape.Length != 2)
            {
                throw new InvalidOperationException("Need to specify parameters for tensors with more than 2 dims.");
            }
            return Transpose(0, 1, inline: inline);
        }

        public IntTensor Transpose(int dimension1, int dimension2, IntTensor result = null, bool inline = false)
        {
            if (!IsContiguous())
            {
                throw new InvalidOperationException("Tensor must be contiguous, call Contiguous() to convert");
            }

            if (dimension1 < 0 || dimension1 >= shape.Length)
                throw new ArgumentOutOfRangeException("dimension1");
            if (dimension2 < 0 || dimension2 >= shape.Length)
                throw new ArgumentOutOfRangeException("dimension2");

            if (dimension1 == dimension2)
            {
                return this;
            }

            var newShape = (int[])shape.Clone();
            var tmpDimension = newShape[dimension1];
            newShape[dimension1] = newShape[dimension2];
            newShape[dimension2] = tmpDimension;

            result = factory.Create(newShape);

            var nCpu = SystemInfo.processorCount;
            Parallel.For(0, nCpu, workerId =>
            {
                var max = size * (workerId + 1) / nCpu;
                for (var i = size * workerId / nCpu; i < max; i++)
                {
                    var idxs = GetIndices(i);
                    var tmp = idxs[dimension1];
                    idxs[dimension1] = idxs[dimension2];
                    idxs[dimension2] = tmp;
                    result[idxs] = this[i];
                }
            });
            return result;
        }

        public IntTensor createZerosTensorLike() {
            IntTensor new_tensor = this.emptyTensorCopy ();
            new_tensor.Zero_ ();
            return new_tensor;
        }

        public IntTensor createOnesTensorLike() {
            IntTensor new_tensor = this.emptyTensorCopy();
            new_tensor.Zero_ ();
            new_tensor.Add ((int)1,true);
            return new_tensor;
        }

        public IntTensor Div(IntTensor x, bool inline = false, IntTensor result = null)
        {
            if (!IsContiguous() || !x.IsContiguous()) {
                throw new InvalidOperationException ("Tensor must be contiguous, call Contiguous() to convert");
            }

            // Check if both tensors are compatible for sub - fallback to scalar version if either tensor's size == 1
            if (SameSizeDimensionsShapeAndLocation(ref x))
            {
                if (x.Size == 1)
                {
                    return this.Div(x.Expand(shape).Contiguous(), inline);
                }
                else if (this.Size == 1)
                {
                    if (inline)
                    {
                        throw new InvalidOperationException("Tensor sizes don't match");
                    }

                    return x.Div(this.Expand(x.shape).Contiguous()).Pow(-1);
                }
                else
                {
                    throw new InvalidOperationException();
                }
            }
            result = HookGraph(ref result, tensor_inputs:new IntTensor[]{x}, creation_op:"div_elem", inline:inline);
            
            if (dataOnGpu & x.dataOnGpu)
            {
                result.Gpu(shader);
                if (inline)
                {
                    if (autograd)
                        throw new InvalidOperationException(
                            "Cannot call inline functions if you intend to run backprop.");
                    DivElemGPU_(x);
                    return this;
                }
                result = DivElemGPU(x, result);
            }
            else
            {
                result.Data = data.AsParallel().Zip(x.Data.AsParallel(), (a, b) => a / b).ToArray();
            }

            return result;
        }

        public IntTensor Div(int value, bool inline = false, IntTensor result = null)
        {
            result = HookGraph (ref result, scalar_input:value, creation_op:"div_scalar", inline:inline);
            
            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (!inline) return DivScalarGPU(value, result);
                DivScalarGPU_(value);
                return this;
            }
            result.Data = data.AsParallel().Select(x => x / value).ToArray();
            return result;
        }

        public bool Equal(IntTensor x, bool inline = false)
        {
            if (dataOnGpu)
            {
                throw new NotImplementedException();
            }

            return this.Shape.SequenceEqual(x.Shape) && data.AsParallel().SequenceEqual(x.Data.AsParallel());
        }


        public IntTensor MM(IntTensor x, IntTensor result = null)
        {
            if (!IsContiguous() || !x.IsContiguous()) {
                throw new InvalidOperationException ("All tensors must be contiguous, call Contiguous() to convert");
            }

            if (this.shape.Length != 2 || x.shape.Length != 2)
            {
                throw new InvalidOperationException(
                    "Cannot do MM on tensors that aren't 2 dimentional. Try calling view() to reshape");
            }
            
            result = HookGraph( result:ref result, 
                                tensor_inputs:new IntTensor[]{x},  
                                creation_op:"mm", 
                                inline:false, 
                                resultShape:new int[]{shape[0],x.shape[1]});
            
            result.AddMatrixMultiply(this, x);

            return result;
        }

        public IntTensor Mul(IntTensor x, bool inline = false, IntTensor result = null)
        {
            if (!IsContiguous() || !x.IsContiguous()) {
                throw new InvalidOperationException ("All tensors must be contiguous, call Contiguous() to convert");
            }

            // Check if both tensors are compatible for sub - fallback to scalar version if either tensor's size == 1
            if (SameSizeDimensionsShapeAndLocation(ref x))
            {
                if (x.Size == 1)
                {
                    return this.Mul(x.Expand(shape).Contiguous(), inline);
                }
                else if (this.Size == 1)
                {
                    if (inline)
                    {
                        throw new InvalidOperationException("Tensor sizes don't match");
                    }

                    return x.Mul(this.Expand(x.shape).Contiguous());
                }
                else
                {
                    throw new InvalidOperationException();
                }
            }

            result = HookGraph(ref result, tensor_inputs: new IntTensor[]{x}, creation_op:"mul_elem", inline:inline);

            if (dataOnGpu && x.dataOnGpu)
            {
                if (inline)
                {
                    if (autograd)
                    {
                        throw new InvalidOperationException(
                            "Cannot call inline functions if you intend to run backprop.");
                    }
                    MulElemGPU_(x);
                    return this;
                }
                result = MulElemGPU(x, result);
            }
            else
            {
                result.Data = data.AsParallel().Zip(x.Data.AsParallel(), (a, b) => a * b).ToArray();
            }

            return result;
        }

        public IntTensor Mul(int value, bool inline = false, IntTensor result = null)
        {
            result = HookGraph (ref result,  creation_op: "mul_scalar", inline:inline, scalar_input:value);

            if (dataOnGpu)
            {
                if (!inline) return MulScalarGPU(value, result);
                MulScalarGPU_(value);
                return this;
            }

            result.Data = data.AsParallel().Select(x => x * value).ToArray();
            return result;
        }


        public IntTensor Sub(IntTensor x, bool inline = false)
        {

            IntTensor result = factory.Create(this.shape);

            if (dataOnGpu)
            {
                result.Gpu(shader);
                if (inline) { SubGPU_(x); return this; }
                else { return SubGPU(x, result); }
            }
            else
            {
                result = inline ? this : factory.Create(this.shape);
                // run Subtraction on the CPU
                result.Data = data.AsParallel().Zip(x.Data.AsParallel(), (a, b) => a - b).ToArray();

                return result;
            }

        }

        public IntTensor Pow(IntTensor x, bool inline = false, IntTensor result = null)
        {
            if (!IsContiguous() || !x.IsContiguous())
            {
                throw new InvalidOperationException("All tensors must be contiguous, call Contiguous() to convert");
            }

            result = inline ? this : factory.Create(this.shape);

            result.Data = data.AsParallel().Zip(
              x.Data.AsParallel(),
              (a, b) => (int)Math.Pow((int)a, b)
            ).ToArray();

            return result;
        }

        public IntTensor Pow(int value, bool inline = false, IntTensor result = null)
        {
            result = inline ? this : factory.Create(this.shape);

            result.Data = data.AsParallel().Select(x => (int)Math.Pow(x, value)).ToArray();

            return result;
        }

        public IntTensor Sub(int value, bool inline = false)
        {
            if (dataOnGpu)
            {
                throw new NotImplementedException();
            }

            IntTensor result = inline ? this : factory.Create(this.shape);
            result.Data = data.AsParallel().Select(x => x - value).ToArray();

            return result;
        }

        public IntTensor Tan(bool inline = false)
        {
            if (dataOnGpu)
            {
                throw new NotImplementedException();
            }
            IntTensor result = factory.Create(this.shape);
            result.Data = data.AsParallel().Select(x => (int)Math.Tan((int)x)).ToArray();
            return result;
        }

        public int Trace()
        {
            if ((shape.Length != 2) || (shape[0] != shape[1]))
                throw new InvalidOperationException("Trace is defined on square 2d matrices only.");

            if (dataOnGpu)
            {
                throw new NotImplementedException();
            }

            var stride = strides[0] + strides[1];
            return Enumerable.Range(0, shape.Min()).AsParallel().Select(i => this[i * stride]).Sum();
        }

        // closes class and namespace
    }
}
