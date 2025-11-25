import streamlit as st
import izracun_kolicine
import izracun_sestavin
import kalkulator_kolicine
import kalkulator_viskoznosti_kolicine


st.sidebar.title("izberi program ki ga želite uporabiti")
choice=st.sidebar.radio ( "programi",["izračun količine","izračun sestavin","kalkulator količine","kalkulator viskoznosti in količin"])
if choice=="izračun količine":
    izracun_kolicine.run()
elif choice=="izračun sestavin":
    izracun_sestavin.run()
elif choice=="kalkulator količine":
    kalkulator_kolicine.run() 
elif choice=="kalkulator viskoznosti in količin":
    kalkulator_viskoznosti_kolicine.run()



