// Drowsiness API
// Tested
const imageInput = document.getElementById("image-input");
const imageContainer = document.getElementById("image-container");
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");

imageInput.addEventListener("change", async function(event)  {
    let file = await event.target.files[0];
    let imageURL = await URL.createObjectURL(file);   // What does this line mean ?
    let image = new Image();
    image.src = await imageURL;
    image.onload = async function() {
        ctx.drawImage(image, 0, 0, 640, 480);
        const base64Data = canvas.toDataURL("image/jpeg");

        let drowsiness_score = await inference(base64Data);
        console.log(drowsiness_score);
    }
})

const loadDrowsinessModel = (async () => {
    const model = tf.loadGraphModel("model/model.json");
    return model;
});

const preprocess = (source, modelWidth, modelHeight) => {
    let xRatio, yRatio; // ratios for boxes

    const input = tf.tidy(() => {
      
      const img = tf.browser.fromPixels(source);
  
      // padding image to square => [n, m] to [n, n], n > m
      const [h, w] = img.shape.slice(0, 2); // get source width and height
      const maxSize = Math.max(w, h); // get max size
      const imgPadded = img.pad([
        [0, maxSize - h], // padding y [bottom only]
        [0, maxSize - w], // padding x [right only]
        [0, 0],
      ]);
  
      xRatio = maxSize / w; // update xRatio
      yRatio = maxSize / h; // update yRatio
  
      return tf.image
        .resizeBilinear(imgPadded, [modelWidth, modelHeight]) // resize frame
        .div(255.0) // normalize
        .expandDims(0); // add batch
    });
  
    return [input, xRatio, yRatio];
  };

// The model takes in img base 64 and return drowsiness score.
async function drowsinessPostprocess(pixels){
    const model = await loadDrowsinessModel();
    let res = model.predict(pixels);
    res = res.transpose([0, 2, 1]);
    const rawScores = res.slice([0, 0, 4], [-1, -1, -1]).squeeze(0);
    let {values, indices} = tf.topk(rawScores.max(1), 20);
    let selected_detections = tf.gather(res, indices, axis = 1);
    let selected_boxes = selected_detections.slice([0, 0, 0], [-1, -1, 4]).squeeze(0);
    let selected_scores = selected_detections.slice([0, 0, 4], [-1, -1, -1]).max(-1).squeeze(0);
    let selected_classes = selected_detections.slice([0, 4, 4], [-1, -1, -1]).argMax(-1).squeeze(0);
    let nms = await tf.image.nonMaxSuppressionAsync(selected_boxes, selected_scores, 1, 0.6, 0.4);

    let output_boxes = selected_boxes.gather(nms, 0).dataSync();
    let output_scores = selected_scores.gather(nms, 0).dataSync();
    let output_classes = selected_classes.gather(nms, 0).dataSync();
    let drowsiness_score = 0;

    if(output_classes.length == 0){
      return drowsiness_score;
    }
    else{
      if(output_classes[0] == 0){
        drowsiness_score = 1 - output_scores[0];
      }
      else{
        drowsiness_score = output_scores[0];
      }
    }

    return drowsiness_score;
}

async function inference(img_base64)
{
    if(typeof img_base64 !== "string" || !img_base64.startsWith('data:image')){
        throw new Error('Invalid input: the image must be in base64 format.')
    }

    const imageElement = document.createElement("img");

    imageElement.src = img_base64;
    await imageElement.decode();
    let [pixels, xRatio, yRatio] = await preprocess(imageElement, 640, 640);
    let drowsiness_score = await drowsinessPostprocess(pixels);
    // Make it only detect the person in the middle of the image

    return  drowsiness_score;
}



