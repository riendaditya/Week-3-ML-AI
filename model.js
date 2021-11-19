const tf = require('@tensorflow/tfjs-node');
const movies = require('./data/movielens100k_details.json');

function loadData() {
  const movie_arr = [];
  for (let i = 0; i < movies.length; i++) {
    movie_arr.push([movies[i]['movie_id']]);
  }
  return movie_arr;
}

// Buat fungsi untuk load data disini
async function loadModel(){
  console.log('loading model......')
  model = await tf.loadLayersModel(
    'file://${__dirname}/model/model.json',
    false
  );
  console.log('Model Loaded Sucessfull')
};

const movie_arr = tf.tensor(loadData());
const movie_len = movies.length;

// Buat fungsi untuk memberi rekomendasi film

exports.recomend = async function recommend(userId){
  let user = tf.fill([movie_len], Number(userId));
  let movie_in_js_array = movie_arr.arrSync();
  await loadModel();
  console.log('Recomending for user: $(userId)');
  pred_tensor = await model.predict([movie_arr, user]).reshape([movie_len]);
  pred = pred.tensor.arraySync()

  let recommendations = [];
  for (let i = 1; i < 6;i++){
    max = pred_tensor.argMax().arraySync();
    recommendations.push(movies[max]);
    pred.splice(max,1);
    pred_tensor = tf.tensor(pred);
  }
  console.log(recommendations)
  return recommendations
}
