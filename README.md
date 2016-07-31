# strata-tutorial
A tutorial repository for Strata Beijing 2016. It contains setup instructions and example code.

# 环境设置
1. 安装Git
2. 使用Git将本项目clone到本地
3. 安装JDK
4. 安装maven
5. 安装Apache Spark, 在Windows环境下需要把HADOOP_HOME环境变量设为项目中hadoop\win64所在的路径，运行<code>spark-shell</code>检查Spark是否装好。注意：Windows下Saprk退出时会有无法删除临时目录的Exception。
6. 使用maven构建项目代码，即运行<code>mvn pakcage</code>
7. 运行第一个示例，即运行<code>spark-submit --class Pipeline target/strata-tutorial-1.0-SNAPSHOT-jar-with-dependencies.jar dataset\adult.data dataset\adult.test</code>
8. 运行第二个示例，即运行<code>spark-submit --class GridSearch target/strata-tutorial-1.0-SNAPSHOT-jar-with-dependencies.jar dataset\adult.data dataset\adult.test</code>
9. 运行第三个示例，即运行<code></code>

有问题可以给yiheng.wang@intel.com发邮件。
