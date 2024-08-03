# Customer Segmentation (Unsupervised Machine Learning for Cluster Analysis)

## About The Project 
<div style="font-family: Arial; font-size: 12pt; font-weight: bold; line-height:1.5"> 
  
**This project employs unsupervised machine learning for cluster analysis and customer segmentation. The dataset used here is comprised of thousands of records of customer purchases and shopping habits. The goal of this project is, first, to analyze and understand in depth the customer base present in the dataset, and, second, to utilize machine learning algorithms for cluster analysis in order to breakdown the customer base into distinct clusters customer groups. Cluster analysis can be very useful for understanding a customer base and aiding businesses to tailor targeted marketing strategies, optimize their product offerings, and be better able to meet their customers' needs and enhance their shopping experience. As such, after segmenting the customer base into separate groups, the groups will then be analyzed and compared to develop a thorough understanding of the different customer groups, their characteristics, preferences, shopping habits and their needs. This subsequently will guide efforts to curate targeted marketing campaigns, develop customer retention strategies, and/or enhance overall customer satisfaction and loyalty. <br>
In order to perform cluster analysis to segment customers into different clusters, first the data was inspected, engineered, and processed in preparation for analysis and modeling. After preparing the data, four clustering algorithms were developed and evaluated in order to find the most suitable algorithm for task. Having obtained the best clustering model for the data, customers were segmented into groups and the resultant customer groups were analyzed in depth. A report was written describing the findings and identifying the main characteristics or unique features of each customer group, as well as describing the overall similarities and dissimilarities between the customer groups. On the basis of this report, a subsequent section was developed laying out the key insights and takeaways of the cluster analysis as well as providing recommendations to improve sales or curate better marketing campaigns tailored to each customer group separately.** <br>
<br>
<br>
</div>

<div style="font-family: Arial; font-size: 12pt; font-weight: bold; line-height:1.5">

<strong> **Overall, this project is broken down into 7 sections:** </strong> <br>
&emsp;&ensp; **1. Reading and Inspecting the Data** <br>
&emsp;&ensp; **2. Updating the Data** <br>
&emsp;&ensp; **3. Exploratory Data Analysis** <br>
&emsp;&ensp; **4. Data Preprocessing** <br>
&emsp;&ensp; **5. Model Development and Evaluation (Cluster Analysis)** <br>
&emsp;&ensp; **6. Model Interpretation** <br>
&emsp;&ensp; **7. Key Insights and Recommendations**
</div>
<br>
<br>


## About The Data  
<div style="font-family: Arial; font-size: 12pt; font-weight: bold; line-height:1.5">

**The present dataset was taken from Kaggle.com, a popular platform for finding and publishing datasets. You can quickly access it by clicking [here](https://www.kaggle.com/datasets/zeesolver/consumer-behavior-and-shopping-habits-dataset/data). The dataset consists of around 4,000 records of customer purchases. For each entry here, a customer is assigned a unique identifier and their purchase, preferences, and other relevant details are recorded. Indeed, the dataset encompasses a wide variety of variables, including demographic information about the customers, their shopping frequency and purchase history, their product preferences and overall satisfaction with the product purchased. This, therefore, makes the current dataset ideal analyzing and understanding consumer behavior, decision-making, and for the purposes of cluster analysis and customer segmentation.**
<br> 
<br>

**You can view each column and its description in the table below:** <br> </div> 

| **Variable**      | **Description**                                                                                         |
| :-----------------| :------------------------------------------------------------------------------------------------------ |
| **Customer ID**   | Unique identifier for each customer       |
| **Age**           | Age of the customer                       |
| **Gender**        | Gender of the customer                    |
| **Item Purchased** | Item or product purchased              |
| **Category**      | Category of the item purchased (e.g., clothing, accessory, etc.)          |
| **Purchase Amount (USD)**| Amount spent (in USD) in a given transaction |
| **Location**      | Location from which a purchase was made         |
| **Size**          | Size of the purchased item (if applicable) |
| **Color**         | Color of the purchased item or product     |
| **Season**        | Season in which the item was purchased (e.g., winter, spring, etc.)  |
| **Review Rating** | Rating score given by a customer for the item purchased (on a 5-point rating scale) |
| **Subscription Status** | Indicates whether or not a customer is subscribed to the brand or shop service |
| **Shipping Type** | Method of delivery or shipping type (e.g., standard shipping, express, store pickup, etc.) |
| **Discount Applied** | Indicates whether or not a discount was applied to the purchase  |
| **Promo Code Used** | Indicates whether or or not a promo code or coupon was used during purchase        |
| **Previous Purchases** | Number of prior purchases made by the same customer  |
| **Payment Method** | Method of payment for the purchase (e.g., cash, credit card, paypal, etc.)     |
| **Frequency of Purchases** | Frequency of engagement of a customer in purchasing activities (e.g., weekly, monthly, annually, etc.)   |


<br>
<br>

**Here's a sample of the dataset being analyzed:**
<br> 

<img src="shopping customers screenshot.jpg" alt="https://github.com/Mo-Khalifa96/Customer-Segmentation/blob/main/shopping%20customers%20screenshot.jpg" width="800"/>

<br>
<br>

## Quick Access 
<div style="font-family: Arial; font-size: 12pt; font-weight: bold; line-height:1.5">

**To quickly access the project, I provided two links, both of which will direct you to Jupyter Notebook with all the code and corresponding output rendered and organized into separate sections and sub-sections. Each section is provided with thorough explanations that guide the project development one step at a time. The first link, however, only allows you to view the project without interacting with its code. The second link allows you to both the view the code and also interact with it directly to reproduce the analysis results if you wish so. To execute the code, please make sure to run the first two cells first in order to install the Python modules necessary for the task and make them ready for use. To run any given block of code, simply select the cell and click on the 'Run' icon on the notebook toolbar.** </div>
<br>
<br>
<br>

***To view the project only, click on the following link:*** <br>
https://nbviewer.org/github/Mo-Khalifa96/Customer-Segmentation/blob/main/Customer%20Segmentation%20%28Unsupervised%20ML%20for%20Cluster%20Analysis%29.ipynb
<br>
<br>
***Alternatively, to view the project and interact with its code, click on the following link:*** <br>
https://mybinder.org/v2/gh/Mo-Khalifa96/Customer-Segmentation/main?labpath=Customer+Segmentation+%28Unsupervised+ML+for+Cluster+Analysis%29.ipynb
<br>
</div>
<br>


