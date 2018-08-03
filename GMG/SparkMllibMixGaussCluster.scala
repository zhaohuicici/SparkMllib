package SparkMllibOtheralgorithm



import org.apache.spark.mllib.clustering.{GaussianMixture, KMeans}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

/**
  * 高斯混合聚类
  * 高斯分布：当一个数据向量在一个高斯分布的模型计算与之以内，则认为它与高斯分布相匹配，属于此模型的聚类．
  * 混合高斯分布：任何样本的聚类都可以使用多个单高斯分布模型来表示．
  *GMG
  * https://my.oschina.net/sunmin/blog/722845
  *
  */
object SparkMllibMixGaussCluster {
  val conf = new SparkConf()                                     //创建环境变量
    .setMaster("local")                                             //设置本地化处理
    .setAppName("gaussian")                              //设定名称
  val sc = new SparkContext(conf)

  def main(args: Array[String]) {
    val data = sc.textFile("./src/SparkMllibOtheralgorithm/gmg.txt")
    val parsedData = data.map(s => Vectors.dense(s.trim().split(' ').map(_.toDouble)))
      .cache()

    val model = new GaussianMixture().setK(2).run(parsedData) // 设置训练模型的分类数
    for (i <- 0 until model.k) {
      println("weight=%f\nmu=%s\nsigma=\n%s\n" format			//逐个打印单个模型
        (model.weights(i), model.gaussians(i).mu, model.gaussians(i).sigma))	//打印结果
    }
  }
}
