package SparkMllibOtheralgorithm


import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  *  MLLib分类，逻辑回归，是分类，不是回归
  *  胃癌转移判断
  * https://my.oschina.net/sunmin/blog/719742
  */
object SparkMllibLogisticRegression {
  val conf = new SparkConf() //创建环境变量
    .setMaster("local")      //设置本地化处理
    .setAppName("LogisticRegression4")//设定名称
  val sc = new SparkContext(conf)

  def main(args: Array[String]) {
    val data = MLUtils.loadLibSVMFile(sc, "./src/SparkMllibOtheralgorithm/wa.txt")	//读取数据文件,一定注意文本格式
    val splits = data.randomSplit(Array(0.7, 0.3), seed = 11L)	//对数据集切分
    val parsedData = splits(0)		//分割训练数据
    val parseTtest = splits(1)		//分割测试数据
    val model = LogisticRegressionWithSGD.train(parsedData,50)	//训练模型

    val predictionAndLabels = parseTtest.map {//计算测试值
      case LabeledPoint(label, features) =>	//计算测试值
        val prediction = model.predict(features)//计算测试值
        (prediction, label)			//存储测试和预测值
    }

    val metrics = new MulticlassMetrics(predictionAndLabels)//创建验证类
    val precision = metrics.precision			//计算验证值
    println("Precision = " + precision)	//打印验证值

    val patient = Vectors.dense(Array(70,3,180.0,4,3))	//计算患者可能性
    if(patient == 1) println("患者的胃癌有几率转移。")//做出判断
    else println("患者的胃癌没有几率转移。")	//做出判断
    //Precision = 0.3333333333333333
    //患者的胃癌没有几率转移。
  }
}
