# Neural Style Transfer (NST) - README

## 專案簡介
本專案為 Neural Style Transfer (NST) 的實驗與實作報告，目的是結合一張圖片的內容與另一張圖片的風格，產生具有藝術感的新圖片。

專案名稱：**Neural Style Transfer (NST)**  
作者：李晉方 (學號：111062240)

---

## 方法與使用工具
- **程式語言**：Python
- **深度學習框架**：PyTorch
- **預訓練模型**：VGG-19
- **優化器**：Adam Optimizer
- **影像處理**：PIL、torchvision

---

## 實驗流程

1. **影像預處理**
   - 將輸入圖片尺寸調整至與內容圖片相同
   - 依據 ImageNet 標準進行正規化處理

2. **特徵擷取**
   - 使用 VGG-19 特定層擷取內容特徵與風格特徵

3. **損失計算**
   - **內容損失**：內容圖片與生成圖片在特徵圖上的均方誤差 (MSE)
   - **風格損失**：風格圖片與生成圖片在 Gram matrix 上的均方誤差 (MSE)

4. **圖片生成**
   - 以內容圖片初始化生成圖片
   - 進行 5000 次迭代，持續優化最小化總損失 (內容損失 + 風格損失)

5. **結果可視化**
   - 每隔一段迭代步驟儲存中間生成結果，觀察生成過程

---

## 數學細節
- **內容損失 (Content Loss)**
- **風格損失 (Style Loss)**：透過 Gram matrix 計算特徵的共變異資訊
- **總損失 (Total Loss)**：
  \[ \text{Total Loss} = \alpha \times \text{Content Loss} + \beta \times \text{Style Loss} \]

(詳細數學推導與公式可參考報告或 HackMD 上的程式碼)

👉 HackMD 原始碼連結：[HackMD - NST 實作](https://hackmd.io/@jacinto5940304/NST)

---

## 結果展示
- **內容圖片**：海浪照片 (Ocean Wave)
- **風格圖片**：《神奈川沖浪裏》The Great Wave off Kanagawa
- 每 1000 次迭代儲存一次中間生成圖片，觀察漸進式融合過程。

### 訓練過程
- Iteration 1000
- Iteration 2000
- Iteration 3000
- Iteration 4000
- Final result

最終生成圖成功結合了原始內容結構與藝術風格特徵。

---

## 結論
- **內容損失**確保了生成圖保持原圖的主要結構與細節。
- **風格損失**使得生成圖呈現風格圖的紋理與色調。
- 本次實驗加深了我對於 NST 與深度學習技術的理解。
- 同時也熟悉了調整 alpha/beta 權重、進行 iterative optimization、以及中途觀察模型表現的實作經驗。
