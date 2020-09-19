import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

object WordCounter {

  /**从文件中获取单词，按空格分割
   * @param sc SparkContext
   * @param file 文件路径
   * @return RDD[String]
   * 按空格分割单词
   * 单词忽略大小写，这里统一转换为小写
   * @author crx -- 2020.09.10
   */
  def getWords(sc: SparkContext, file: String): RDD[String] = {
    val words: RDD[String] = sc.textFile(file, 1)
    words.flatMap(word => word.split(" ")).map(word => word.toLowerCase)
  }

  def main(args: Array[String]): Unit = {

    val sparkConf = new SparkConf().setAppName("WordCounter").setMaster("local")
    val sc = new SparkContext(sparkConf)

    val url: String = "/Users/chengrongxin/Downloads/第一次作业DDL2020年09月14日/TASK1/test.txt" // 作业的test.txt文件路径

    val word_pool: RDD[(String, Int)] = getWords(sc, url).map(word => (word, 1)). //将单词转换为(word, nums)的键值对
      reduceByKey(_+_).sortBy(_._2)// 相同的单词，次数相加，并按照单词出现频数排序（升）

    word_pool.foreach(word => println(word._1 + ":" + word._2)) // 输出

    sc.stop()

  }
}
