package main

import (
	"errors"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
	"github.com/tidwall/gjson"
)

func setupRouter() *gin.Engine {
	r := gin.Default()
	// basic auth
	godotenv.Load() // it's OK if no .env, as we read from ENV variables instead
	onnxLib := os.Getenv("onnx_lib")
	onnxModel := os.Getenv("onnx_model")
	if len(onnxLib) == 0 {
		panic("onnx_lib not set")
	}
	if len(onnxModel) == 0 {
		panic("onnx_model not set")
	}

	r.POST("/api/predict", handleInferencePost)

	return r
}

type InferencePostBody struct {
	Id               string
	TimeStamp        time.Time
	Waveform         [][]float32
	Sensitivity      float32
	ReturnPrediction bool
}

func parseJson(data string) (*InferencePostBody, error) {
	id := gjson.Get(data, "id").String()
	if len(id) == 0 {
		return nil, errors.New("id not found")
	}
	timestampRaw := gjson.Get(data, "timestamp").String()
	if len(timestampRaw) == 0 {
		return nil, errors.New("timestamp not found")
	}
	timestamp, err := time.Parse("2006-01-02T15:04:05.000000", timestampRaw)
	if err != nil {
		return nil, err
	}
	waveformRaw := gjson.Get(data, "waveform").Array()
	if len(waveformRaw) == 0 {
		return nil, errors.New("waveform not found")
	}
	waveform := make([][]float32, len(waveformRaw))
	for i := range waveformRaw {
		waveformRawArray := waveformRaw[i].Array()
		waveform[i] = make([]float32, len(waveformRawArray))
		for j := range waveformRawArray {
			waveform[i][j] = float32(waveformRawArray[j].Float())
		}
	}

	sensitivityRaw := gjson.Get(data, "sensitivity")
	if !sensitivityRaw.Exists() {
		return nil, errors.New("sensitivity not found")
	}
	sensitivity := float32(sensitivityRaw.Float())

	returnPredictionRaw := gjson.Get(data, "return_prediction")
	if !returnPredictionRaw.Exists() {
		return nil, errors.New("return_prediction not found")
	}

	postContent := InferencePostBody{
		Id:               id,
		TimeStamp:        timestamp,
		Waveform:         waveform,
		Sensitivity:      sensitivity,
		ReturnPrediction: returnPredictionRaw.Bool(),
	}
	return &postContent, nil
}

func handleInferencePost(c *gin.Context) {
	dataRaw, err := io.ReadAll(c.Request.Body)
	if err != nil {
		return
	}
	data := string(dataRaw)
	postContent, err := parseJson(data)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": err.Error(),
		})
		return
	}

	// do preprocessing
	windpwLength := 4800
	hopLength := 2400
	wave := padZeroTransform(postContent.Waveform, windpwLength, hopLength)
	waveBatched := waveformToBatchTransform(wave, windpwLength, hopLength)
	waveBatched = batchNormalizeTransform(waveBatched)

	// do inference
	onnxLib := os.Getenv("onnx_lib")
	onnxModel := os.Getenv("onnx_model")
	var predictionBatched [][][]float32
	for _, batch := range waveBatched {
		// Reshape the batch to have the first dimension as 1 as batch
		batchReshaped := make([][][]float32, 1)
		batchReshaped[0] = batch

		prediction, err := inference(batchReshaped, onnxLib, onnxModel)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": err.Error(),
			})
			return
		}

		// Append the prediction to the predictionBatched
		predictionBatched = append(predictionBatched, prediction[0])
	}
	prediction := batchToWaveformTransform(predictionBatched, windpwLength, hopLength)

	// extract peaks for prediction[1] as P wave, prediction[2] as S wave, prediction[3] as PS
	// also ignore two peaks within 1s assuming sampling rate is 40
	peaksP, hieghtsP := findPeaks(prediction[1], postContent.Sensitivity, 40)
	peaksS, hieghtsS := findPeaks(prediction[2], postContent.Sensitivity, 40)
	peaksPS, heightsPS := findPeaks(prediction[3], postContent.Sensitivity, 40)

	// with reference to timeStamp, calculate timeP and timeS
	timeP := make([]time.Time, len(peaksP))
	timeS := make([]time.Time, len(peaksS))
	timePS := make([]time.Time, len(peaksPS))
	for i := range peaksP {
		timeP[i] = postContent.TimeStamp.Add(time.Duration(peaksP[i]) * time.Second / 40)
	}
	for i := range peaksS {
		timeS[i] = postContent.TimeStamp.Add(time.Duration(peaksS[i]) * time.Second / 40)
	}
	for i := range peaksPS {
		timePS[i] = postContent.TimeStamp.Add(time.Duration(peaksPS[i]) * time.Second / 40)
	}

	// return result
	ReturnArrivals := gin.H{
		"P": gin.H{
			"peaks":   peaksP,
			"heights": hieghtsP,
			"times":   timeP,
		},
		"S": gin.H{
			"peaks":   peaksS,
			"heights": hieghtsS,
			"times":   timeS,
		},
		"PS": gin.H{
			"peaks":   peaksPS,
			"heights": heightsPS,
			"times":   timePS,
		},
	}

	ReturnPrediction := gin.H{
		"noise": prediction[0],
		"P":     prediction[1],
		"S":     prediction[2],
		"PS":    prediction[3],
	}

	if postContent.ReturnPrediction {
		c.JSON(http.StatusOK, gin.H{
			"id":         postContent.Id,
			"arrivals":   ReturnArrivals,
			"prediction": ReturnPrediction,
		})
	} else {
		c.JSON(http.StatusOK, gin.H{
			"id":       postContent.Id,
			"arrivals": ReturnArrivals,
		})
	}
}

func main() {
	gin.SetMode(gin.ReleaseMode)

	listenAddr := ":" + os.Getenv("PORT")

	r := setupRouter()
	r.Run(listenAddr)
}
