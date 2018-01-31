using System;
using NUnit.Framework;
using OpenMined.Network.Controllers;
using OpenMined.Protobuf;
using UnityEngine;
using Google.Protobuf;
using OpenMined.Protobuf.Onnx;

namespace OpenMined.Tests.Editor.IntTensorTests
{
    [Category("IntTensorCPUTests")]
    public class IntTensorCPUTest
    {
        private SyftController ctrl;

        [OneTimeSetUp]
        public void Init()
        {
            //Init runs once before running test cases.
            ctrl = new SyftController(null);
        }

        [OneTimeTearDown]
        public void CleanUp()
        {
            //CleanUp runs once after all test cases are finished.
        }

        [SetUp]
        public void SetUp()
        {
            //SetUp runs before all test cases
        }

        [TearDown]
        public void TearDown()
        {
            //SetUp runs after all test cases
        }

        /********************/
        /* Tests Start Here */
        /********************/

        [Test]
        public void Abs()
        {
            int[] shape1 = { 2, 5 };
            int[] data1 = { -1, -2, -3, -4, 5, 6, 7, 8, -999, 10 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] expectedData1 = { 1, 2, 3, 4, 5, 6, 7, 8, 999, 10 };
            int[] shape2 = { 2, 5 };
            var expectedTensor1 = ctrl.intTensorFactory.Create(_data: expectedData1, _shape: shape2);

            var actualTensorAbs1 = tensor1.Abs();

            for (int i = 0; i < actualTensorAbs1.Size; i++)
            {
                Assert.AreEqual(expectedTensor1[i], actualTensorAbs1[i]);
            }
        }

        [Test]
        public void Abs_()
        {
            int[] shape1 = { 2, 5 };
            int[] data1 = { -1, -2, -3, -4, 5, 6, 7, 8, -999, 10 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] expectedData1 = { 1, 2, 3, 4, 5, 6, 7, 8, 999, 10 };
            int[] shape2 = { 2, 5 };
            var expectedTensor1 = ctrl.intTensorFactory.Create(_data: expectedData1, _shape: shape2);

            var actualTensorAbs1 = tensor1.Abs(inline: true);

            for (int i = 0; i < actualTensorAbs1.Size; i++)
            {
                Assert.AreEqual(expectedTensor1[i], actualTensorAbs1[i]);
            }
        }

        [Test]
        public void Add()
        {
            float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = { 3, 2, 6, 9, 10, 1, 4, 8, 5, 7 };
            int[] shape2 = { 2, 5 };
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var tensorSum = tensor1.Add(tensor2);

            for (int i = 0; i < tensorSum.Size; i++)
            {
                Assert.AreEqual(tensor1[i] + tensor2[i], tensorSum[i]);
            }
        }

        [Test]
        public void Add_()
        {
            float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = { 3, 2, 6, 9, 10, 1, 4, 8, 5, 7 };
            int[] shape2 = { 2, 5 };
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            float[] data3 = { 4, 4, 9, 13, 15, 7, 11, 16, 14, 17 };
            int[] shape3 = { 2, 5 };
            var tensor3 = ctrl.floatTensorFactory.Create(_data: data3, _shape: shape3);

            tensor1.Add(tensor2, inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(tensor3[i], tensor1[i]);
            }
        }

		[Test]
		public void Neg()
		{
			int[] shape1 = { 2, 5 };
			int[] data1 = { -1, -2, -3, -4, 5, 6, 7, 8, -999, 10 };
			var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

			int[] expectedData1 = { 1, 2, 3, 4, -5, -6, -7, -8, 999, -10 };
			int[] shape2 = { 2, 5 };
			var expectedTensor1 = ctrl.intTensorFactory.Create(_data: expectedData1, _shape: shape2);

			var actualTensorNeg1 = tensor1.Neg();

			for (int i = 0; i < actualTensorNeg1.Size; i++)
			{
				Assert.AreEqual(expectedTensor1[i], actualTensorNeg1[i]);
			}
		}

		[Test]
		public void Neg_()
		{
			int[] shape1 = { 2, 5 };
			int[] data1 = { -1, -2, -3, -4, 5, 6, 7, 8, -999, 10 };
			var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

			int[] expectedData1 = { 1, 2, 3, 4, -5, -6, -7, -8, 999, -10 };
			int[] shape2 = { 2, 5 };
			var expectedTensor1 = ctrl.intTensorFactory.Create(_data: expectedData1, _shape: shape2);

			tensor1.Neg(inline: true);

			for (int i = 0; i < tensor1.Size; i++)
			{
				Assert.AreEqual(expectedTensor1[i], tensor1[i]);
			}
		}

        [Test]
        public void Reciprocal()
        {
            int[] data1 = {1, 2, 3, -1};
            int[] shape1 = { 4 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] data2 = {1, 0, 0, -1};
            int[] shape2 = { 4 };
            var expectedTensor = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            var actualTensor = tensor1.Reciprocal();

            for (int i = 0; i < expectedTensor.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], actualTensor[i]);
            }
        }

        [Test]
        public void Reciprocal_()
        {
            int[] data1 = { 1, 2, 3, -1 };
            int[] shape1 = { 4 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] data2 = { 1, 0, 0, -1 };
            int[] shape2 = { 4 };
            var expectedTensor = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            tensor1.Reciprocal(inline: true);

            for (int i = 0; i < expectedTensor.Size; i++)
            {
                Assert.AreEqual(expectedTensor[i], tensor1[i]);
            }
        }

        [Test]
        public void Eq()
        {
            int[] data1 = { 1, 2, 3, 4 };
            int[] shape = { 2, 2 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape);

            int[] data2 = { 1, 2, 1, 2 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape);

            int[] expectedData = { 1, 1, 0, 0 };
            var expectedOutput = ctrl.intTensorFactory.Create(_data: expectedData, _shape: shape);

            var eqOutput = tensor1.Eq(tensor2);

            for (int i = 0; i < expectedOutput.Size; i++)
            {
                Assert.AreEqual(expectedOutput[i], eqOutput[i]);
            }
        }

        [Test]
        public void Eq_()
        {
            int[] data1 = { 1, 2, 3, 4 };
            int[] shape = { 2, 2 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape);

            int[] data2 = { 1, 2, 1, 2 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape);

            int[] expectedData = { 1, 1, 0, 0 };
            var expectedOutput = ctrl.intTensorFactory.Create(_data: expectedData, _shape: shape);

            tensor1.Eq(tensor2, inline:true);

            for (int i = 0; i < expectedOutput.Size; i++)
            {
                Assert.AreEqual(expectedOutput[i], tensor1[i]);
            }
        }

        public void Equal()
        {
            int[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] data2 = { 3, 2, 6, 9, 10, 1, 4, 8, 5, 7 };
            int[] shape2 = { 2, 5 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            var tensor3 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] differentShapedData = { 0, 0 };
            int[] differentShape = { 1, 2 };
            var differentShapedTensor = ctrl.intTensorFactory.Create(_data: differentShapedData, _shape: differentShape);

            Assert.False(tensor1.Equal(differentShapedTensor));
            Assert.False(tensor1.Equal(tensor2));
            Assert.True(tensor1.Equal(tensor3));
        }

        [Test]
        public void PowElem()
        {
            int[] data1 = { 1, 2, 3, 4, 5, 1, 2, 3, 4, 5 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] data2 = { 5, 4, 3, 2, 1, 1, 2, 3, 4, 5 };
            int[] shape2 = { 2, 5 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            int[] data3 = { 1, 16, 27, 16, 5, 1, 4, 27, 256, 3125 };
            int[] shape3 = { 2, 5 };
            var tensor3 = ctrl.intTensorFactory.Create(_data: data3, _shape: shape3);

            var result = tensor1.Pow(tensor2);

            for (int i = 0; i < result.Size; i++)
            {
                Assert.AreEqual(tensor3[i], result[i]);
            }
        }

        [Test]
        public void PowElem_()
        {
            int[] data1 = { 1, 2, 3, 4, 5, 1, 2, 3, 4, 5 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] data2 = { 5, 4, 3, 2, 1, 1, 2, 3, 4, 5 };
            int[] shape2 = { 2, 5 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            int[] data3 = { 1, 16, 27, 16, 5, 1, 4, 27, 256, 3125 };
            int[] shape3 = { 2, 5 };
            var tensor3 = ctrl.intTensorFactory.Create(_data: data3, _shape: shape3);

            tensor1.Pow(tensor2, inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(tensor3[i], tensor1[i]);
            }
        }

        [Test]
        public void PowScalar()
        {
            int[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] data2 = { 1, 4, 9, 16, 25, 36, 49, 64, 81, 100 };
            int[] shape2 = { 2, 5 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            var result = tensor1.Pow(2);

            for (int i = 0; i < result.Size; i++)
            {
                Assert.AreEqual(tensor2[i], result[i]);
            }
        }

        [Test]
        public void PowScalar_()
        {
            int[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] data2 = { 1, 4, 9, 16, 25, 36, 49, 64, 81, 100 };
            int[] shape2 = { 2, 5 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            tensor1.Pow(2, inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(tensor2[i], tensor1[i]);
            }
        }


        [Test]
        public void SplitSameBySize()
        {
            float[] data = {1, 2, 3, 4, 5, 6, 7, 8};
            int[] shape = {2, 4};

            var tensor = ctrl.intTensorFactory.Create(_data: data, _shape: shape);

            var splits = tensor.Split(1);
            Assert.AreEqual(2, splits.Length);

            for(int i = 0; i < splits.Length; i++)
            {
                Assert.AreEqual(2, splits[i].Shape.Length);
            }

            int[] expectedShape = {1, 4};

            for(int i = 0; i < splits.Length; i++){
                for(int j = 0; j < splits[0].Shape.Length; j++)
                {
                    Assert.AreEqual(expectedShape[j], splits[i].Shape[j]);
                }
            }

            float[] splitData1 = {1, 2, 3, 4};
            float[] splitData2 = {5, 6, 7, 8};

            var expectedTensor1 = ctrl.intTensorFactory.Create(_data: splitData1, _shape: expectedShape);
            var expectedTensor2 =   ctrl.intTensorFactory.Create(_data: splitData2, _shape: expectedShape);

            for(int i = 0; i < expectedShape[0]; i++){
                for(int j = 0; j < expectedShape[1]; j++)
                {   
                    Assert.AreEqual(expectedTensor1[i, j], splits[0][i, j]);
                    Assert.AreEqual(expectedTensor2[i, j], splits[1][i, j]);
                }

            }

            var splits2 = tensor.Split(2, dim:1);
            Assert.AreEqual(2, splits2.Length);

            for(int i = 0; i < splits2.Length; i++)
            {
                Assert.AreEqual(2, splits2[i].Shape.Length);
            }

            int[] expectedShape2 = {2, 2};

            for(int i = 0; i < splits2.Length; i++)
            {
                for(int j = 0; j < splits2[0].Shape.Length; j++)
                {
                    Assert.AreEqual(expectedShape2[j], splits2[i].Shape[j]);
                }
            }

            float[] splitData3 = {1, 2, 5, 6};
            float[] splitData4 = {3, 4, 7, 8};
            
            var expectedTensor3 = ctrl.intTensorFactory.Create(_data: splitData3, _shape: expectedShape2);
            var expectedTensor4 = ctrl.intTensorFactory.Create(_data: splitData4, _shape: expectedShape2);

            for(int i = 0; i < expectedShape2[0]; i++){
                for(int j = 0; j < expectedShape2[1]; j++)
                {   
                    Assert.AreEqual(expectedTensor3[i, j], splits2[0][i, j]);
                    Assert.AreEqual(expectedTensor4[i, j], splits2[1][i, j]);
                }
            }
        }

        [Test]
        public void SplitDiffBySize()
        {
            float[] data = {1, 2, 3, 4, 5, 6, 7, 8};
            int[] shape = {2, 4};

            var tensor = ctrl.intTensorFactory.Create(_data: data, _shape: shape);
            
            var splits = tensor.Split(3, 1);
            Assert.AreEqual(2, splits.Length);

            for(int i = 0; i < splits.Length; i++)
            {
                Assert.AreEqual(2, splits[i].Shape.Length);
            }

            int[] expectedShape1 = {2, 3};
            int[] expectedShape2 = {2, 1};

            
            for(int j = 0; j < splits[0].Shape.Length; j++)
            {
                Assert.AreEqual(expectedShape1[j], splits[0].Shape[j]);
                Assert.AreEqual(expectedShape2[j], splits[1].Shape[j]);
            }
            
            float[] splitData1 = {1, 2, 3, 5, 6, 7};
            float[] splitData2 = {4, 8};

            var expectedTensor1 = ctrl.intTensorFactory.Create(_data: splitData1, _shape: expectedShape1);
            var expectedTensor2 =  ctrl.intTensorFactory.Create(_data: splitData2, _shape: expectedShape2);

            for(int i = 0; i < expectedShape1[0]; i++){
                for(int j = 0; j < expectedShape1[1]; j++)
                {   
                    Assert.AreEqual(expectedTensor1[i, j], splits[0][i, j]);
                }
            }

            for(int i = 0; i < expectedShape2[0]; i++){
                for(int j = 0; j < expectedShape2[1]; j++)
                {   
                    Assert.AreEqual(expectedTensor2[i, j], splits[1][i, j]);
                }
            }
        }

        [Test]
        public void SplitSameBySections()
        {
            float[] data = {1, 2, 3, 4, 5, 6, 7, 8};
            int[] shape = {2, 4};

            var tensor = ctrl.intTensorFactory.Create(_data: data, _shape: shape);

            int[] sections = {1,1};
            var splits = tensor.Split(sections);
            Assert.AreEqual(2, splits.Length);

            for(int i = 0; i < splits.Length; i++)
            {
                Assert.AreEqual(2, splits[i].Shape.Length);
            }

            int[] expectedShape = {1, 4};

            for(int i = 0; i < splits.Length; i++){
                for(int j = 0; j < splits[0].Shape.Length; j++)
                {
                    Assert.AreEqual(expectedShape[j], splits[i].Shape[j]);
                }
            }

            float[] splitData1 = {1, 2, 3, 4};
            float[] splitData2 = {5, 6, 7, 8};

            var expectedTensor1 = ctrl.intTensorFactory.Create(_data: splitData1, _shape: expectedShape);
            var expectedTensor2 =   ctrl.intTensorFactory.Create(_data: splitData2, _shape: expectedShape);

            for(int i = 0; i < expectedShape[0]; i++){
                for(int j = 0; j < expectedShape[1]; j++)
                {   
                    Assert.AreEqual(expectedTensor1[i, j], splits[0][i, j]);
                    Assert.AreEqual(expectedTensor2[i, j], splits[1][i, j]);
                }

            }

            int[] sections2 = {2,2};
            var splits2 = tensor.Split(sections2, dim:1);
            Assert.AreEqual(2, splits2.Length);

            for(int i = 0; i < splits2.Length; i++)
            {
                Assert.AreEqual(2, splits2[i].Shape.Length);
            }

            int[] expectedShape2 = {2, 2};

            for(int i = 0; i < splits2.Length; i++)
            {
                for(int j = 0; j < splits2[0].Shape.Length; j++)
                {
                    Assert.AreEqual(expectedShape2[j], splits2[i].Shape[j]);
                }
            }

            float[] splitData3 = {1, 2, 5, 6};
            float[] splitData4 = {3, 4, 7, 8};
            
            var expectedTensor3 = ctrl.intTensorFactory.Create(_data: splitData3, _shape: expectedShape2);
            var expectedTensor4 = ctrl.intTensorFactory.Create(_data: splitData4, _shape: expectedShape2);

            for(int i = 0; i < expectedShape2[0]; i++){
                for(int j = 0; j < expectedShape2[1]; j++)
                {   
                    Assert.AreEqual(expectedTensor3[i, j], splits2[0][i, j]);
                    Assert.AreEqual(expectedTensor4[i, j], splits2[1][i, j]);
                }
            }

        }

        [Test]
        public void SplitDiffBySections()
        {
            float[] data = {1, 2, 3, 4, 5, 6, 7, 8};
            int[] shape = {2, 4};

            var tensor = ctrl.intTensorFactory.Create(_data: data, _shape: shape);

            int[] sections = {3, 1};
            var splits = tensor.Split(sections, 1);
            Assert.AreEqual(2, splits.Length);

            for(int i = 0; i < splits.Length; i++)
            {
                Assert.AreEqual(2, splits[i].Shape.Length);
            }

            int[] expectedShape1 = {2, 3};
            int[] expectedShape2 = {2, 1};

            
            for(int i = 0; i < splits[0].Shape.Length; i++)
            {
                Assert.AreEqual(expectedShape1[i], splits[0].Shape[i]);
                Assert.AreEqual(expectedShape2[i], splits[1].Shape[i]);
            }
            
            float[] splitData1 = {1, 2, 3, 5, 6, 7};
            float[] splitData2 = {4, 8};

            var expectedTensor1 = ctrl.intTensorFactory.Create(_data: splitData1, _shape: expectedShape1);
            var expectedTensor2 =  ctrl.intTensorFactory.Create(_data: splitData2, _shape: expectedShape2);

            for(int i = 0; i < expectedShape1[0]; i++){
                for(int j = 0; j < expectedShape1[1]; j++)
                {   
                    Assert.AreEqual(expectedTensor1[i, j], splits[0][i, j]);
                }
            }

            for(int i = 0; i < expectedShape2[0]; i++){
                for(int j = 0; j < expectedShape2[1]; j++)
                {   
                    Assert.AreEqual(expectedTensor2[i, j], splits[1][i, j]);
                }
            }
        }

        [Test]
        public void SplitSizeLargerThanDim()
        {
            float[] data = {1, 2, 3, 4};
            int[] shape = {1, 4};

            var tensor = ctrl.intTensorFactory.Create(_data: data, _shape: shape);
            
            var splits = tensor.Split(3);

            Assert.AreEqual(1, splits.Length);
            Assert.AreEqual(2, splits[0].Shape.Length);

            for(int i = 0; i < splits[0].Shape.Length; i++)
            {
                Assert.AreEqual(shape[i], splits[0].Shape[i]);
            }

            for(int i = 0; i < shape[0]; i++){
                for(int j = 0; j < shape[1]; j++)
                {   
                    Assert.AreEqual(tensor[i, j], splits[0][i, j]);
                }
            }
        }

        [Test]
        public void SplitSectionsWithZero()
        {
            float[] data = {1, 2, 3, 4, 5, 6};
            int[] shape = {2, 3};

            var tensor = ctrl.intTensorFactory.Create(_data: data, _shape: shape);
            
            int[] sections = {1,0,2};
            var splits = tensor.Split(sections, 1);

            Assert.AreEqual(3, splits.Length);

            for(int i = 0; i < splits.Length; i++)
            {
                Assert.AreEqual(2, splits[i].Shape.Length);
            }

            int[] expectedShape1 = {2, 1};
            int[] expectedShape2 = {2, 0};
            int[] expectedShape3 = {2, 2};

            
            for(int i = 0; i < splits[0].Shape.Length; i++)
            {
                Assert.AreEqual(expectedShape1[i], splits[0].Shape[i]);
                Assert.AreEqual(expectedShape2[i], splits[1].Shape[i]);
                Assert.AreEqual(expectedShape3[i], splits[2].Shape[i]);
            }

            float[] splitData1 = {1, 4};
            float[] splitData3 = {2, 3, 5, 6};

            var expectedTensor1 = ctrl.intTensorFactory.Create(_data: splitData1, _shape: expectedShape1);
            var expectedTensor3 =   ctrl.intTensorFactory.Create(_data: splitData3, _shape: expectedShape3);

            for(int i = 0; i < expectedShape1[0]; i++){
                for(int j = 0; j < expectedShape1[1]; j++)
                {   
                    Assert.AreEqual(expectedTensor1[i, j], splits[0][i, j]);
                }
            }

            Assert.AreEqual(0, splits[1].Data.Length);

            for(int i = 0; i < expectedShape3[0]; i++){
                for(int j = 0; j < expectedShape3[1]; j++)
                {   
                    Assert.AreEqual(expectedTensor3[i, j], splits[2][i, j]);
                }
            }
        }

        [Test]
        public void Sqrt()
        {
            int[] data1 = {1, 4, 9, 16};
            int[] shape1 = {4};

            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);
            var result = tensor1.Sqrt();

            int[] data2 = {1, 2, 3, 4};
            int[] shape2 = {4};
            var expectedTensor = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            for (int i = 0; i < tensor1.Data.Length; i++)
            {
                Assert.AreEqual(expectedTensor[i], result[i], 1e-3);
            }
        }

        [Test]
        public void Sub()
        {
            int[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] data2 = { 3, 2, 6, 9, 10, 1, 4, 8, 5, 7 };
            int[] shape2 = { 2, 5 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            var tensorDiff = tensor1.Sub(tensor2);

            for (int i = 0; i < tensorDiff.Size; i++)
            {
                Assert.AreEqual(tensor1[i] - tensor2[i], tensorDiff[i]);
            }
        }

        [Test]
        public void Sub_()
        {
            int[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] data2 = { 3, 2, 6, 9, 10, 1, 4, 8, 5, 7 };
            int[] shape2 = { 2, 5 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            int[] data3 = { -2,  0, -3, -5, -5,  5,  3,  0,  4,  3 };
            int[] shape3 = { 2, 5 };
            var tensor3 = ctrl.intTensorFactory.Create(_data: data3, _shape: shape3);

            tensor1.Sub(tensor2, inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(tensor3[i], tensor1[i]);
            }
        }

        [Test]
        public void SubScalar()
        {
            int[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int scalar = 5;

            var tensorDiff = tensor1.Sub(scalar);

            for (int i = 0; i < tensorDiff.Size; i++)
            {
                Assert.AreEqual(tensor1[i] - scalar, tensorDiff[i]);
            }
        }

        [Test]
        public void SubScalar_()
        {
            int[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int scalar = 5;

            int[] data2 = { -4, -3, -2, -1,  0,  1,  2,  3,  4,  5 };
            int[] shape2 = { 2, 5 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            tensor1.Sub(scalar, inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(tensor2[i], tensor1[i]);
            }
        }

        [Test]
        public void Sign()
        {
            int[] data1 = {-1,2,3,-5,6,-10};
            int[] shape1 = {2,3};
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] data2 = {-1,1,1,-1,1,-1};
            int[] shape2 = {2,3};
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            var tensor3 = tensor1.Sign(inline: false);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(tensor2[i], tensor3[i]);
            }
        }

        [Test]
        public void Tan()
        {
            float[] data1 = {30, 20, 40, 50};
            int[] shape1 = {4};
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = {-6.4053312f, 2.23716094f, -1.11721493f, -0.27190061f};
            int[] shape2 = {4};
            var expectedTanTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var actualTanTensor = tensor1.Tan();

            for (int i = 0; i < actualTanTensor.Size; i++)
            {
                Assert.AreEqual(expectedTanTensor[i], actualTanTensor[i]);
            }
        }

        [Test]
        public void Trace()
        {
            // test #1
            int[] data1 = {2, 2, 3, 4};
            int[] shape1 = {2, 2};
            var tensor = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);
            int actual = tensor.Trace();
            int expected = 6;

            Assert.AreEqual(expected, actual);

            // test #2
            int[] data3 = {1, 2, 3};
            int[] shape3 = {3};
            var non2DTensor = ctrl.intTensorFactory.Create(_data: data3, _shape: shape3);
            Assert.That(() => non2DTensor.Trace(),
                Throws.TypeOf<InvalidOperationException>());
        }

        [Test]
        public void Lt()
        {
            int[] data1 = { 1, 2, 3, 4 };
            int[] shape = { 2, 2 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape);

            int[] data2 = { 2, 2, 1, 2 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape);

            int[] expectedData = { 1, 0, 0, 0 };
            var expectedOutput = ctrl.intTensorFactory.Create(_data: expectedData, _shape: shape);

            var ltOutput = tensor1.Lt(tensor2);

            for (int i = 0; i < expectedOutput.Size; i++)
            {
                Assert.AreEqual(expectedOutput[i], ltOutput[i]);
            }
        }

        [Test]
        public void Lt_()
        {
            int[] data1 = { 1, 2, 3, 4 };
            int[] shape = { 2, 2 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape);

            int[] data2 = { 2, 2, 1, 2 };
            var tensor2 = ctrl.intTensorFactory.Create(_data: data2, _shape: shape);

            int[] expectedData = { 1, 0, 0, 0 };
            var expectedOutput = ctrl.intTensorFactory.Create(_data: expectedData, _shape: shape);

            tensor1.Lt(tensor2, inline:true);

            for (int i = 0; i < expectedOutput.Size; i++)
            {
                Assert.AreEqual(expectedOutput[i], tensor1[i]);
            }
        }

        [Test]
        public void Sin()
        {
            int[] data1 = { 15, 60, 90, 180 };
            int[] shape1 = { 4 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = { 0.65028784f, -0.30481062f, 0.89399666f, -0.80115264f };
            int[] shape2 = { 4 };
            var expectedSinTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var actualSinTensor = tensor1.Sin();

            for (int i = 0; i < actualSinTensor.Size; i++)
            {
                Assert.AreEqual(expectedSinTensor[i], actualSinTensor[i]);
            }
        }

        [Test]
        public void Cos()
        {
            int[] data1 = { 30, 60, 90, 180 };
            int[] shape1 = { 4 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = { 0.1542515f, -0.952413f, -0.4480736f, -0.5984601f };
            int[] shape2 = { 4 };
            var expectedCosTensor = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var actualCosTensor = tensor1.Cos();

            for (int i = 0; i < actualCosTensor.Size; i++)
            {
                Assert.AreEqual(expectedCosTensor[i], actualCosTensor[i], 0.00001f);
            }

        }

        [Test]
        public void GetProto()
        {
            int[] data = {-1, 0, 1, int.MaxValue, int.MinValue};
            int[] shape = {5};
            Syft.Tensor.IntTensor t = ctrl.intTensorFactory.Create(_data: data, _shape: shape);

            TensorProto message = t.GetProto();
            byte[] messageAsByte = message.ToByteArray();
            TensorProto message2 = TensorProto.Parser.ParseFrom(messageAsByte);

            Assert.AreEqual(message, message2);
        }
        [Test]
        public void view()
        {
            int[] data1 = { 4, 4, 7, 7, 2, 2, 4, 8, 7, 8, 8, 0, 6, 2, 8, 9 };
            int[] shape1 = { 2, 2, 4 };
            var tesnor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] data2 = { 4, 4, 7, 7, 2, 2, 4, 8, 7, 8, 8, 0, 6, 2, 8, 9 };
            int[] shape2 = { 8, 2 };
            var expectedIntTesnor = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            var actualIntTensor = tesnor1.View(shape2);
            for(int i = 0; i < actualIntTensor.Size; i++)
            {
                Assert.AreEqual(expectedIntTesnor[i], actualIntTensor[i]);
            }
        }

        [Test]
        public void view_()
        {
            int[] data1 = { 4, 4, 7, 7, 2, 2, 4, 8, 7, 8, 8, 0, 6, 2, 8, 9 };
            int[] shape1 = { 2, 2, 4 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] data2 = { 4, 4, 7, 7, 2, 2, 4, 8, 7, 8, 8, 0, 6, 2, 8, 9 };
            int[] shape2 = { 8, 2 };
            var expectedIntTesnor = ctrl.intTensorFactory.Create(_data: data2, _shape: shape2);

            tensor1.View(shape2, inline: true);
            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(expectedIntTesnor[i], tensor1[i]);
            }
        }

        /* closes class and namespace */
    }
}
