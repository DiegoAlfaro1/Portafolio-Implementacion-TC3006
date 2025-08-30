'''
Codigo Auxiliar creado por claude para decidir si es necesario agregar los tipos de cafe al modelo
'''

# ANALYSIS: Impact of Coffee Type on Premium Sales
print(f"\n ANALISIS DEL IMPACTO DEL TIPO DE CAFE:")

# Premium rate by coffee type
coffee_premium_analysis = df.groupby('Coffe_name').agg({
    'Premium_sale': ['count', 'sum', 'mean'],
    'Price': ['min', 'max', 'mean']
}).round(3)

coffee_premium_analysis.columns = ['Total_Ventas', 'Ventas_Premium', 'Tasa_Premium', 'Precio_Min', 'Precio_Max', 'Precio_Promedio']
coffee_premium_analysis = coffee_premium_analysis.sort_values('Tasa_Premium', ascending=False)

print("Analisis por tipo de cafe:")
print(coffee_premium_analysis)

# Calculate potential improvement
best_predictor_accuracy = df.groupby('Coffe_name')['Premium_sale'].apply(
    lambda x: max(x.mean(), 1-x.mean())  # Best possible accuracy if we just predict majority class per coffee
).mean()

print(f"\nPrecision potencial usando solo tipo de cafe: {best_predictor_accuracy:.1%}")
print(f"Precision actual (temporal): {resultados['test_accuracy']:.1%}")
print(f"Mejora potencial: +{best_predictor_accuracy - resultados['test_accuracy']:.1%}")

# Visualization of coffee type impact
plt.figure(figsize=(12, 8))
coffee_viz_data = coffee_premium_analysis.reset_index()

plt.subplot(2, 2, 1)
sns.barplot(data=coffee_viz_data, x='Tasa_Premium', y='Coffe_name', palette='viridis')
plt.title('Tasa de Ventas Premium por Tipo de Café')
plt.xlabel('Proporción de Ventas Premium')

plt.subplot(2, 2, 2)
sns.barplot(data=coffee_viz_data, x='Total_Ventas', y='Coffe_name', palette='plasma')
plt.title('Volumen de Ventas por Tipo de Café')
plt.xlabel('Número Total de Ventas')

plt.subplot(2, 2, 3)
sns.scatterplot(data=coffee_viz_data, x='Precio_Promedio', y='Tasa_Premium', 
                size='Total_Ventas', sizes=(50, 500), alpha=0.7)
for i, row in coffee_viz_data.iterrows():
    plt.annotate(row['Coffe_name'], (row['Precio_Promedio'], row['Tasa_Premium']), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)
plt.title('Precio vs Tasa Premium por Café')
plt.xlabel('Precio Promedio')
plt.ylabel('Tasa Premium')

plt.subplot(2, 2, 4)
# Current model vs coffee-type-only prediction
comparison_data = pd.DataFrame({
    'Método': ['Modelo Actual\n(Temporal)', 'Potencial\n(Tipo Café)'],
    'Precisión': [resultados['test_accuracy'], best_predictor_accuracy]
})
sns.barplot(data=comparison_data, x='Método', y='Precisión', palette=['lightcoral', 'lightgreen'])
plt.title('Comparación de Precisión')
plt.ylabel('Precisión del Modelo')
plt.ylim(0, 1)

plt.tight_layout()
plt.show()