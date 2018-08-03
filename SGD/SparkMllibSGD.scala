package SparkMllibOtheralgorithm



/**
  * 随机梯度下降法（stochastic gradient descent，SGD）
  * SGD是最速梯度下降法的变种。
  * 使用最速梯度下降法，将进行N次迭代，直到目标函数收敛，或者到达某个既定的收敛界限。
  * 每次迭代都将对m个样本进行计算，计算量大。
  * 为了简便计算，SGD每次迭代仅对一个样本计算梯度，直到收敛。
  * 随机梯度下降，即（最快速从紫金山山顶下去）
  *
  * https://my.oschina.net/sunmin/blog/719638
  */
import scala.collection.mutable.HashMap

object SparkMllibSGD {
  val data = HashMap[Int,Int]()	//创建数据集
  def getData():HashMap[Int,Int] = {//生成数据集内容
    for(i <- 1 to 50){	//创建50个数据
      data += (i -> (16*i))//写入公式y=16x
    }
    data		//返回数据集
  }

  var θ:Double = 0	//第一步假设θ为0
  var α:Double = 0.1	//设置步进系数，每次下降的幅度大小

  def sgd(x:Double,y:Double) = {//设置迭代公式
    θ = θ - α * ( (θ*x) - y)	//迭代公式
  }
  def main(args: Array[String]) {
    val dataSource = getData()	//获取数据集
    dataSource.foreach(myMap =>{//开始迭代
      sgd(myMap._1,myMap._2)//输入数据
    })
    println("最终结果θ值为 " + θ)//显示结果
  }
}
