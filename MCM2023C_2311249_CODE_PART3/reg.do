import excel "/Users/ljs/Desktop/美赛C/Problem_C_Data_Wordle_edited.xlsx", sheet("Sheet1") firstrow clear
 
rename try try1
rename tries try2
rename I try3
rename J try4
rename K try5
rename L try6

 
gen diff = (try1+try2+try3+try4>70)
gen hard_ratio = Numberinhardmode/Numberofreportedresults
tsset Contestnumber
gen hard_ratiofor = L.hard_ratio

gen dif1 = (difficultyLevel >0)
gen dif2 = (difficultyLevel >1)
gen dif3 = (difficultyLevel >2)


gen aver_try = (try1+try2+try3+try4+try5+try6)/6
gen sq_try = ((try1-aver_try)^2+(try2-aver_try)^2+(try3-aver_try)^2+(try4-aver_try)^2+(try5-aver_try)^2+(try6-aver_try)^2)/6


reg hard_ratio frequencySum wordVar frequencyGap frequencyGap2 frequencyLevel repeatLetters Location Contestnumber diff Numberofreportedresults


reg Numberofreportedresults difficultyScores logDate

reg Numberofreportedresults dif1 logDate

reg Numberofreportedresults dif2 logDate

reg Numberofreportedresults dif3 logDate


save "data.dta",replace
