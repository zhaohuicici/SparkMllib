package SparkMlibCourse5

import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkContext, SparkConf}

/**
  * 朴素贝叶斯僵尸粉鉴定(朴素贝叶斯需要非负特征值)
  * 正常用户标记为１，虚假用户标记为０
  * V(v1,v2,v3)
  * v1 = 已发微博/注册天数
  * v2 = 好友数量/注册天数
  * v3 = 是否有手机
  * 已发微博/注册天数　< 0.05, V1 = 0
  * 0.05 <= 已发微博/注册天数　< 0.75, V1 = 1
  * 0.75 <= 已发微博/注册天数, V1 = 2
  * https://my.oschina.net/sunmin/blog/720089
  */


/*
*
* 优点
对待预测样本进行预测，过程简单速度快(想想邮件分类的问题，预测就是分词后进行概率乘积，在log域直接做加法更快)。
对于多分类问题也同样很有效，复杂度也不会有大程度上升。
在分布独立这个假设成立的情况下，贝叶斯分类器效果奇好，会略胜于逻辑回归，同时我们需要的样本量也更少一点。
对于类别类的输入特征变量，效果非常好。对于数值型变量特征，我们是默认它符合正态分布的。
缺点
对于测试集中的一个类别变量特征，如果在训练集里没见过，直接算的话概率就是0了，预测功能就失效了。当然，我们前面的文章提过我们有一种技术叫做『平滑』操作，可以缓解这个问题，最常见的平滑技术是拉普拉斯估测。
那个…咳咳，朴素贝叶斯算出的概率结果，比较大小还凑合，实际物理含义…恩，别太当真。
朴素贝叶斯有分布独立的假设前提，而现实生活中这些predictor很难是完全独立的。
最常见应用场景
文本分类/垃圾文本过滤/情感判别：这大概会朴素贝叶斯应用做多的地方了，即使在现在这种分类器层出不穷的年 代，在文本分类场景中，朴素贝叶斯依旧坚挺地占据着一席之地。原因嘛，大家知道的，因为多分类很简单，同时在文本数据中，分布独立这个假设基本是成立的。 而垃圾文本过滤(比如垃圾邮件识别)和情感分析(微博上的褒贬情绪)用朴素贝叶斯也通常能取得很好的效果。
多分类实时预测：这个是不是不能叫做场景？对于文本相关的多分类实时预测，它因为上面提到的优点，被广泛应用，简单又高效。
推荐系统：是的，你没听错，是用在推荐系统里！！朴素贝叶斯和协同过滤(Collaborative Filtering)是一对好搭档，协同过滤是强相关性，但是泛化能力略弱，朴素贝叶斯和协同过滤一起，能增强推荐的覆盖度和效果。
运行代码如下
* */
object SparkMlibBayesAdd1 {
  val conf = new SparkConf()   //创建环境变量
    .setMaster("local")        //设置本地化处理
    .setAppName("ZombieBayes") //设定名称
  val sc = new SparkContext(conf)

  def main(args: Array[String]) {
   // val data = sc.textFile("C:/Users/cici/Desktop/SparkAdvanceCourse/src/SparkMlibCourse5/data.txt")
    val data = sc.textFile("./src/SparkMlibCourse5/data.txt")

    //C:\Users\cici\Desktop\SparkAdvanceCourse\src\SparkMlibCourse5
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }

    val splits = parsedData.randomSplit(Array(0.7, 0.3), seed = 11L)			//对数据进行分配
    val trainingData = splits(0)									//设置训练数据
    val testData = splits(1)									//设置测试数据
    val model = NaiveBayes.train(trainingData, lambda = 1.0)			//训练贝叶斯模型
    val predictionAndLabel = testData.map(p => (model.predict(p.features), p.label)) //验证模型
    val accuracy = 1.0 * predictionAndLabel.filter(					//计算准确度
      label => label._1 == label._2).count()						//比较结果
    println(accuracy)
    val test = Vectors.dense(0, 0, 10)
    val result = model.predict(test)//预测一个特征　
    println(result)//2
  }
}
