package main

import (
	"errors"

	ort "github.com/yalue/onnxruntime_go"
)

func inference(inputData [][][]float32, onnxLib string, onnxModel string) ([][][]float32, error) {
	ort.SetSharedLibraryPath(onnxLib)

	err := ort.InitializeEnvironment()
	if err != nil {
		return nil, err
	}
	defer ort.DestroyEnvironment()

	numBatches, numChannels, numSamples := len(inputData), len(inputData[0]), len(inputData[0][0])
	// numChannels should be 3
	if numChannels != 3 {
		return nil, errors.New("numChannels should be 3")
	}
	if numBatches != 1 {
		return nil, errors.New("currently only support numBatches=1")
	}
	inputDataFlat := make([]float32, numBatches*numChannels*numSamples)
	for i := range inputData {
		for j := range inputData[i] {
			for k := range inputData[i][j] {
				inputDataFlat[i*numChannels*numSamples+j*numSamples+k] = inputData[i][j][k]
			}
		}
	}

	inputShape := ort.NewShape(int64(numBatches), int64(numChannels), int64(numSamples))
	inputTensor, err := ort.NewTensor(inputShape, inputDataFlat)
	if err != nil {
		return nil, err
	}
	defer inputTensor.Destroy()

	outputShape := ort.NewShape(int64(numBatches), 4, int64(numSamples))
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, err
	}
	defer outputTensor.Destroy()

	session, err := ort.NewSession[float32](onnxModel,
		[]string{"waveform"}, []string{"prediction"},
		[]*ort.Tensor[float32]{inputTensor}, []*ort.Tensor[float32]{outputTensor})
	if err != nil {
		return nil, err
	}
	defer session.Destroy()

	err = session.Run()
	if err != nil {
		return nil, err
	}

	outputData := outputTensor.GetData()

	// unflatten outputData as shape (numBatches, numClasses, numSamples)
	outputDataUnflattened := make([][][]float32, numBatches)
	for i := range outputDataUnflattened {
		outputDataUnflattened[i] = make([][]float32, 4)
		for j := range outputDataUnflattened[i] {
			outputDataUnflattened[i][j] = make([]float32, numSamples)
			for k := range outputDataUnflattened[i][j] {
				outputDataUnflattened[i][j][k] = outputData[i*4*numSamples+j*numSamples+k]
			}
		}
	}

	// do softmax along the numClasses axis
	outputDataUnflattened = softmax(outputDataUnflattened)

	// find peaks along the index 1,2 classes as P peaks and S peaks

	return outputDataUnflattened, nil
}
