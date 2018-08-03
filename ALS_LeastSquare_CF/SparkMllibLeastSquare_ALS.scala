package SparkMllibOtheralgorithm


/**
  *  协同过滤算法，基于 (交替最小二乘法) ALS 计算
  *  人以群分，物以类聚,
  *  ALS是统计分析中最常用的一种逼近计算算法．
  *  输入数据集 Ratings 是　ALS 固定输入格式,
  *  Ratings [Int, Int, Double] 即[用户名，物品名，评分]
  *
  *  https://my.oschina.net/sunmin/blog/719273
  */
package spark.collaborativeFiltering

import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.{SparkConf, SparkContext}

object SparkMllibLeastSquare_ALS {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("ALS").setMaster("local")
    val sc= new SparkContext(conf)
    val data = sc.textFile("./src/SparkMllibOtheralgorithm//ul.txt")
    val ratings = data.map(_.split(" ") match {
      case Array(user, item, rate) =>
        Rating(user.toInt, item.toInt, rate.toDouble)//将数据集转化为专用的 Rating
    })

    val rank = 2 // 模型中隐藏因子数
    val numInterations = 5 //算法迭代次数
    val model = ALS.train(ratings, rank, numInterations, 0.01) // 进行模型训练
    val result = model.recommendProducts(2, 1)//为用户 2 推荐一个商品
    result.foreach(println)
    //Rating(2,15,3.9713808775549495)，为用户 2 推荐一个编号 15 的商品，预测评分 3.97 与实际的 4 接近．
  }
}
