package SparkMllibOtheralgorithm

import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * 随机雨林决策树
  * 若干个决策树组成的决策树森林，
  * 随机雨林的实质就是建立多个决策树，然后取得所有决策树的平均值
  * ps:一个数据集中包括一项评分，假设一共５个分数，在实际应用中采用二分法
  *   1 2 3 | 4 5
  *   即　bin  有２个，分别装有数据集{1,2,3},{4,5}
  *   split被设置为３
  *
  * https://my.oschina.net/sunmin/blog/720717
  */
object SparkMllibRandomForest {
  val conf = new SparkConf()                                     //创建环境变量
    .setMaster("local")                                             //设置本地化处理
    .setAppName("ZombieBayes")                              //设定名称
  val sc = new SparkContext(conf)

  def main(args: Array[String]) {
    val data = MLUtils.loadLibSVMFile(sc, "./src/SparkMllibOtheralgorithm/DTree.txt")

    val numClasses = 2//分类数量
    val categoricalFeaturesInfo = Map[Int, Int]()//设定输入格式
    val numTrees = 3// 随机雨林中决策树的数目
    val featureSubSetStrategy = "auto" //设置属性在节点计算数,自动决定每个节点的属性数
    val impurity = "entropy" //设定信息增益计算方式
    val maxDepth = 5 //最大深度
    val maxBins = 3 // 设定分割数据集

    val model = RandomForest.trainClassifier(
      data,
      numClasses,
      categoricalFeaturesInfo,
      numTrees,
      featureSubSetStrategy,
      impurity,
      maxDepth,
      maxBins
    )// 建立模型

    model.trees.foreach(println)//打印每棵树信息
    println(model.numTrees)
  }
}
