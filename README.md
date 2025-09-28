# Beyond the Flat: What Really Drives HDB Resale Value 
<img src="".\images\pinnacle_logo1.png" height="100" align="right" alt="Pinncale Ventures" style="position:relative; top:-85px;">

<p style="color:#6a737d; font-style:italic;">
  Presented By: Pinnacle Ventures (Alina, Daniel, Leon, Nigel, Rei, Tiffany)
</p>
<img src="C:\Users\Admin\Desktop\Data Sprint\images\hdb_photo.jpg" height = "500" alt="Pinnacle Ventures" >

## Description
HDB resale value isn’t just “near an MRT” anymore—it’s a mix of micro factors (floor area, remaining lease/age, block height, proximity to malls/hawkers and MRT/bus, schools, neighbourhood “affluence”) and macro signals (CPI, GDP, income, demand, month/year, upcoming MRT). Our model blends these into a single prediction of resale_price, grounded in how real buyers value space, access, schools, and timing.

The product gives a price estimate with reasons: which factors added or shaved value (e.g., lease age, transport/amenity proximity, school pressure, MRT development). It’s fast, defensible, and lets agents and buyers make data-backed decisions at a glance.

## Data Dictionary
The following table details the data and their descriptions:
| Column Name	| Description	| Data Type |
| --- | --- | --- | 
| resale_price | the property's sale price in Singapore dollars | Integer |
| demand* | Based off the 1room_sold to studio_apartment_sold and given weightage | Float |
| age_at_sale*  | Taken as transaction year minus lease commencement  | Integer |
| max_floor_lvl | The highest floor in the block | Integer |
| lease_commence_date | When the block’s lease began | Integer |
| Month | Month of the transaction | Integer |
| Tranc_Year  | The year which the transaction happened | Integer |
| affluent_index* | It is the combination of the ranking of the average price by town, ranking the floor category and ranking the flat type. | Float |
| hdb_age  | Number of years from lease_commence_date to present year | Integer |
| floor_area_sqm  | The size of the apartment that was transacted in square meters | Integer |
| amenity_proximity_score* | It is the sum of the average distances between the nearest Mall and Hawker | Float |
| transport_proximity_score* | It is the sum of the average distances between the nearest MRT and Bus Stop | Float |
| amenities_within_1km* | The total number of amenities (Malls and Hawkers) within 1km | Integer |
| avg_subs_sch* | It is the average subscription rate of the schools in the same town | Float |
| num_top_sch* | It is the total number of top schools in the town. And top schools is based off if the subsriptions of a school exceed 200% | Integer |
| mrt_development*| Ranking the development of an MRT station. The rankings are 5 if MRT in development, 4 MRT was built within 2 years, 3 if within 5 years, 2 if within 7 years and 1 for the rest | Integer |
| CPI* | Consumer Price Index by Year | Float |
| GDPM* | Gross Domestic Product by Quarter | Float |
| MHI* | Median Household Income by Quarter | Float|

## Results
Four machine learning models were built: LightGBM, CatBoost, Random Forest, Extra Trees.

The aim of these ML models was to learn from the dataset to predict HDB resale prices.

The following table summarizes the performance of the models:

<table style="border:1px solid #d0d7de; border-collapse:collapse; width:100%;">
  <thead>
    <tr>
      <th style="padding:6px 10px;">Model</th>
      <th style="padding:6px 10px;">Train Accuracy</th>
      <th style="padding:6px 10px;">Test Accuracy</th>
      <th style="padding:6px 10px;">RMSE</th>
      <th style="padding:6px 10px;">Runtime</th>
    </tr>
  </thead>
  <tbody>
<tr style="background:#fff3cd;"> <!-- Light yellow highlight -->
      <td><strong>LightGBM</strong></td>
      <td><strong>99.61%<strong></td><td><strong>97.02%<strong></td><td><strong>S$24,622<strong></td><td><strong>12.5s<strong></td>
    </tr>
    </tr>
    <tr>
      <td style="padding:6px 10px;">CatBoost</td>
      <td style="padding:6px 10px;">97.06%</td>
      <td style="padding:6px 10px;">96.58%</td>
      <td style="padding:6px 10px;">S$26,371</td>
      <td style="padding:6px 10px;">52.3s</td>
    </tr>
    <tr>
      <td style="padding:6px 10px;">Random Forest</td>
      <td style="padding:6px 10px;">99.52%</td>
      <td style="padding:6px 10px;">96.58%</td>
      <td style="padding:6px 10px;">S$26,372</td>
      <td style="padding:6px 10px;">22.9s</td>
    </tr>
        <tr>
      <td style="padding:6px 10px;">Extra Trees</td>
      <td style="padding:6px 10px;">100.00%</td>
      <td style="padding:6px 10px;">96.70%</td>
      <td style="padding:6px 10px;">SS$26,047</td>
      <td style="padding:6px 10px;">4.8s</td>
    </tr>
  </tbody>
</table>



By evaluating the trade-offs between train and test accuracy, model discrepancies, and runtime, LightGBM was been identified as the best model for demonstrating superior performance with no significant overfitting.


## Conclusion
Our results show that remaining lease, distance to MRT, flat size/type and storey, neighbourhood “affluence,” and school pressure are consistent drivers of HDB resale prices, with clear premiums near the CBD and strong MRT access. Macro conditions—inflation (CPI), income (MHI), growth (GDP)—and policy events (e.g., cooling measures) noticeably shift demand and pricing levels.

The model turns these macro-micro signals into a transparent price estimate with explanations, giving agents, buyers, and sellers credible guidance for faster, fairer decisions. As the market evolves, incorporating fresh data (e.g., news or forum sentiment) can further sharpen forecasts and capture shifts in buyer mood.
