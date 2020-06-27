package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"image"
	"image/jpeg"
	_ "image/jpeg"

	"github.com/NOX73/go-neural/learn"
	"github.com/NOX73/go-neural/persist"
	"github.com/nfnt/resize"
)

type Training struct {
	Input  []float64
	Output []float64
}

// Get the bi-dimensional pixel array
func getPixels(img image.Image) ([][]Pixel, error) {

	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	var pixels [][]Pixel
	for y := 0; y < height; y++ {
		var row []Pixel
		for x := 0; x < width; x++ {
			row = append(row, rgbaToPixel(img.At(x, y).RGBA()))
		}
		pixels = append(pixels, row)
	}

	return pixels, nil
}

func getPixelFloat(img image.Image) (result []float64) {

	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			rgba := rgbaToPixel(img.At(x, y).RGBA())

			result = append(result, float64(rgba.R))
			result = append(result, float64(rgba.G))
			result = append(result, float64(rgba.B))

		}

	}

	return
}

// img.At(x, y).RGBA() returns four uint32 values; we want a Pixel
func rgbaToPixel(r uint32, g uint32, b uint32, a uint32) Pixel {
	return Pixel{int(r / 257), int(g / 257), int(b / 257), int(a / 257)}
}

// Pixel struct example
type Pixel struct {
	R int
	G int
	B int
	A int
}

func mains() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	n := persist.FromFile("./learn.json")
	directory := "/Users/arief/go/shopee-product-detection-dataset/test/test/"
	filepath.Walk(directory, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		go func() {
			imagePixel, err := LoadImageFromFile(path)
			if err == nil {
				results := n.Calculate(imagePixel)
				var max float64
				max = 0
				idx := 0
				for index, val := range results {
					if val > max {
						idx = index
						max = val
					}
				}
				log.Println(path, "menghasilkan", idx, "dengan score", results[idx])
			}
		}()
		time.Sleep(1 * time.Nanosecond)
		return nil
	})
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	train()

}
func train() {
	image.RegisterFormat("jpg", "jpg", jpeg.Decode, jpeg.DecodeConfig)

	tempOutput := []float64{
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	}

	var speed float64
	speed = 0.5
	n := persist.FromFile("./learn.json")
	var trainingTotal int
	// n := neural.NewNetwork(120000, []int{2, 2, 42})

	// filename := "0a3d56c9dedaf2858ece21c3d56e56ab.jpg"
	var wg sync.WaitGroup
	for categoryID := 0; categoryID <= 41; categoryID++ {
		// n := persist.FromFile("./learn.json")
		wg.Add(1)
		go func() {
			output := tempOutput
			output[categoryID] = 1
			catString := fmt.Sprintf("%02d", categoryID)
			log.Println(catString)
			directory := "/Users/arief/go/shopee-product-detection-dataset/train/train/" + catString + "/"

			var files []string
			err := filepath.Walk(directory, func(path string, info os.FileInfo, err error) error {
				if err != nil {
					return err
				}
				files = append(files, path)
				return nil
			})
			if err != nil {
				log.Println(err)
			}

			for _, file := range files {

				imagePixel, err := LoadImageFromFile(file)
				if err != nil {
					log.Println(err)
				} else {
					trainingTotal++
					learn.Learn(n, imagePixel, output, speed)
				}
				if trainingTotal%100 == 0 {
					log.Printf("Sudah training %+v", trainingTotal)
				}
			}
			// log.Println("Total training ", trainingTotal)
			persist.ToFile("./learn.json", n)
			wg.Done()
		}()
		time.Sleep(1 * time.Second)
	}
	wg.Wait()
	persist.ToFile("./learn.json", n)
}
func LoadImageFromFile(path string) (pixels []float64, err error) {

	file, err := os.Open(path)

	if err != nil {

		log.Println(path, err)
		return
	}

	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		log.Println(path, err)
		return
	}
	m := resize.Resize(200, 200, img, resize.Lanczos3)

	pixels = getPixelFloat(m)

	if err != nil {
		log.Println(err)
		return
	}
	return
}
