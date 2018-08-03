package SparkMllibOtheralgorithm



import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * 分类 - 朴素贝叶斯简单示例
  * 后验概率　＝　先验概率 ｘ 调整因子
  * https://my.oschina.net/sunmin/blog/720073
  */
object SparkMllibBayes2  {
  val conf = new SparkConf()                                     //创建环境变量
    .setMaster("local")                                             //设置本地化处理
    .setAppName("Bayes")                              //设定名称
  val sc = new SparkContext(conf)

  def main(args: Array[String]) {
    val data = MLUtils.loadLabeledPoints(sc, "./src/SparkMllibOtheralgorithm/bayes.txt")
    val model = NaiveBayes.train(data, 1.0)
    model.labels.foreach(println)//打印　label(labels是标签类别)
    model.pi.foreach(println)//打印先验概率　(pi存储各个label先验概率)
    //0.0
    //1.0
    //2.0
    //-1.0986122886681098
    //-1.0986122886681098
    //-1.0986122886681098
    val test = Vectors.dense(0, 0, 10)//新预测数据
    val result = model.predict(test)//预测结果
    println(result)//2.0
  }
}
