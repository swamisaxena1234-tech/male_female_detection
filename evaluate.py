# Step 5: Evaluate the model
loss, accuracy = model.evaluate(test_set)
print(f"📊 Test Loss: {loss:.4f}")
print(f"🎯 Test Accuracy: {accuracy * 100:.2f}%")
