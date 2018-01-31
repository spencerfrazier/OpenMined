using UnityEngine;

namespace OpenMined.Syft.Tensor
{
    public partial class IntTensor
    {
        // kernel pointers
        [SerializeField] private static int AddElemIntKernel;
        [SerializeField] private static int SubElemIntKernel;
        [SerializeField] private static int SubElemIntKernel_;
        [SerializeField] private static int NegateKernel;
        [SerializeField] private static int ReciprocalIntKernel;
        [SerializeField] private static int ReciprocalIntKernel_;
        [SerializeField] private static int SinIntKernel;
        [SerializeField] private static int CosIntKernel;
        [SerializeField] private static int DivScalarKernel_;
        [SerializeField] private static int DivElemKernel_;
        [SerializeField] private static int DivScalarKernel;
        [SerializeField] private static int DivElemKernel;


        public void initShaderKernels()
        {
            //TODO: This function should only be called once. These members are static!
            if (shader == null) return;
        }

        public IntTensor AddElemGPU(IntTensor tensor, IntTensor result)
        {
            int kernel_id = shader.FindKernel("AddElemInt");

            shader.SetBuffer(kernel_id, "AddElemIntDataA", this.DataBuffer);
            shader.SetBuffer(kernel_id, "AddElemIntDataB", tensor.DataBuffer);
            shader.SetBuffer(kernel_id, "AddElemIntDataResult", result.DataBuffer);

            shader.Dispatch(kernel_id, this.size, 1, 1);

            return result;
        }

        public void AddElemGPU_(IntTensor tensor)
        {
            int kernel_id = shader.FindKernel("AddElemInt_");

            shader.SetBuffer(kernel_id, "AddElemIntDataA_", this.DataBuffer);
            shader.SetBuffer(kernel_id, "AddElemIntDataB_", tensor.DataBuffer);

            shader.Dispatch(kernel_id, this.size, 1, 1);
        }

        public void DivScalarGPU_(float value)
        {
            Debug.LogFormat("<color=blue>IntTensor.DivScalarGPU_ dataOnGpu: {0}</color>", dataOnGpu);

            if (dataOnGpu)
            {
                var valBuffer = SendFloatToGpu(DivScalarKernel_, value, "DivScalarScalar_");

                shader.SetBuffer(DivScalarKernel_, "DivScalarData_", dataBuffer);
                shader.Dispatch(DivScalarKernel_, this.size, 1, 1);

                valBuffer.Release();
            }
        }

        public void DivElemGPU_(IntTensor tensor)
        {
            Debug.LogFormat("<color=blue>IntTensor.DivElemGPU_ dataOnGpu: {0}</color>", dataOnGpu);

            if (dataOnGpu)
            {
                if (tensor.id != this.id) {
                    shader.SetBuffer (DivElemKernel_, "DivElemDataA_", dataBuffer);
                    shader.SetBuffer (DivElemKernel_, "DivElemDataB_", tensor.dataBuffer);
                    shader.Dispatch (DivElemKernel_, this.size, 1, 1);
                }
                else
                {
                    this.ZeroGPU_ ();
                    this.AddScalarGPU_ ((float)1);
                }
            }
        }

        public IntTensor DivScalarGPU(float value, IntTensor result)
        {
            Debug.LogFormat("<color=blue>IntTensor.DivScalarGPU dataOnGpu: {0}</color>", dataOnGpu);

            if (dataOnGpu)
            {
                var valBuffer = SendFloatToGpu(DivScalarKernel, value, "DivScalarScalar");

                shader.SetBuffer(DivScalarKernel, "DivScalarData", dataBuffer);
                shader.SetBuffer(DivScalarKernel, "DivScalarResult", result.dataBuffer);
                shader.Dispatch(DivScalarKernel, this.size, 1, 1);

                valBuffer.Release();
            }
            return result;
        }

        public IntTensor DivElemGPU(IntTensor tensor, IntTensor result)
        {
            Debug.LogFormat("<color=blue>IntTensor.DivElemGPU dataOnGpu: {0}</color>", dataOnGpu);

            if (dataOnGpu)
            {
                if (tensor.id != this.id)
                {
                    shader.SetBuffer(DivElemKernel, "DivElemDataA", dataBuffer);
                    shader.SetBuffer(DivElemKernel, "DivElemDataB", tensor.dataBuffer);
                    shader.SetBuffer(DivElemKernel, "DivElemDataResult", result.dataBuffer);
                    shader.Dispatch(DivElemKernel, this.size, 1, 1);
                }
                else
                {
                    result.ZeroGPU_ ();
                    result.AddScalarGPU_ ((float)1);
                    return result;
                }
            }
            return result;
        }

        public IntTensor ReciprocalGPU(IntTensor result)
        {            
            int kernel_id = shader.FindKernel("ReciprocalInt");

            shader.SetBuffer(kernel_id, "ReciprocalIntData", this.DataBuffer);
            shader.SetBuffer(kernel_id, "ReciprocalIntDataResult", result.DataBuffer);
            shader.Dispatch(kernel_id, this.size, 1, 1);
            return result;
        }

        public void ReciprocalGPU_()
        {
            int kernel_id = shader.FindKernel("ReciprocalInt_");
            shader.SetBuffer(kernel_id, "ReciprocalIntData_", this.DataBuffer);
            shader.Dispatch(kernel_id, this.size, 1, 1);
        }

        public IntTensor SinGPU(IntTensor result)
        {            
            int kernel_id = shader.FindKernel("SinInt");

            shader.SetBuffer(kernel_id, "SinIntData", this.DataBuffer);
            shader.SetBuffer(kernel_id, "SinIntDataResult", result.DataBuffer);
            shader.Dispatch(kernel_id, this.size, 1, 1);
            return result;
        }

        public IntTensor CosGPU(IntTensor result)
        {
            int kernel_id = shader.FindKernel("CosInt");

            shader.SetBuffer(kernel_id, "CosIntData", this.DataBuffer);
            shader.SetBuffer(kernel_id, "CosIntDataResult", result.DataBuffer);
            shader.Dispatch(kernel_id, this.size, 1, 1);
            return result;
        }

        public IntTensor AbsGPU(IntTensor result)
        {
            int kernel_id = shader.FindKernel("AbsElemInt");

            shader.SetBuffer(kernel_id, "AbsElemIntData", this.DataBuffer);
            shader.SetBuffer(kernel_id, "AbsElemIntDataResult", result.DataBuffer);

            shader.Dispatch(kernel_id, this.size, 1, 1);

            return result;
        }

        public void AbsGPU_()
        {
            int kernel_id = shader.FindKernel("AbsElemInt_");
            shader.SetBuffer(kernel_id, "AbsElemIntData_", this.DataBuffer);
            shader.Dispatch(kernel_id, this.size, 1, 1);
        }

        public IntTensor NegGPU(IntTensor result)
        {
            int kernel_id = shader.FindKernel("NegateInt");
            shader.SetBuffer(kernel_id, "NegateIntData", this.DataBuffer);
            shader.SetBuffer(kernel_id, "NegateIntResult", result.DataBuffer);
            shader.Dispatch(kernel_id, this.size, 1, 1);

            return result;
        }

        public void NegGPU_()
        {
            int kernel_id = shader.FindKernel("NegateInt_");
            shader.SetBuffer(kernel_id, "NegateIntData_", this.DataBuffer);
            shader.Dispatch(kernel_id, this.size, 1, 1);
        }

        public IntTensor SubGPU(IntTensor tensor, IntTensor result)
        {
            int kernel_id = shader.FindKernel("SubElemInt");

            shader.SetBuffer(kernel_id, "SubElemIntDataA", this.DataBuffer);
            shader.SetBuffer(kernel_id, "SubElemIntDataB", tensor.DataBuffer);
            shader.SetBuffer(kernel_id, "SubElemIntDataResult", result.DataBuffer);

            shader.Dispatch(kernel_id, this.size, 1, 1);

            return result;
        }

        public void SubGPU_(IntTensor tensor)
        {
            int kernel_id = shader.FindKernel("SubElemInt_");
            shader.SetBuffer(kernel_id, "SubElemIntDataA_", this.DataBuffer);
            shader.SetBuffer(kernel_id, "SubElemIntDataB_", tensor.DataBuffer);

            shader.Dispatch(kernel_id, this.size, 1, 1);
        }

    }
}