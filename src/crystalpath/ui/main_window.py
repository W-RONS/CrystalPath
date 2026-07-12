from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, QTimer, Qt
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDockWidget,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from crystalpath.analysis.interstitials import (
    InterstitialAnalyzer,
    InterstitialResult,
    REFERENCE_ANALYSIS_RADII,
    fractional_position_category,
)
from crystalpath.core.structure import CrystalDocument, site_element, site_occupancy
from crystalpath.rendering.palette import color_for, radius_for
from crystalpath.rendering.viewer import CrystalViewer


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.document: CrystalDocument | None = None
        self.radius_spins: dict[str, QDoubleSpinBox] = {}
        self.analysis_radius_spins: dict[str, QDoubleSpinBox] = {}
        self.interstitial_result: InterstitialResult | None = None
        self._pending_interstitial_visibility_kind: str | None = None
        self._interstitial_visibility_update_scheduled = False
        self.settings = QSettings()
        self.setWindowTitle("CrystalPath V0.3.4")
        self.resize(1380, 820)
        self._build_actions()
        self._build_ui()
        self._build_interstitial_dock()
        self._build_view_toolbar()
        self.statusBar().showMessage("打开一个 CIF 文件开始")

    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        open_action = QAction("打开 CIF…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_cif)
        file_menu.addAction(open_action)

        screenshot_action = QAction("保存视图为 PNG…", self)
        screenshot_action.setShortcut("Ctrl+S")
        screenshot_action.triggered.connect(self.save_screenshot)
        file_menu.addAction(screenshot_action)

        view_menu = self.menuBar().addMenu("视图")
        reset_camera = QAction("重置视角", self)
        reset_camera.setShortcut("R")
        reset_camera.triggered.connect(self.viewer_reset_camera)
        view_menu.addAction(reset_camera)

        analysis_menu = self.menuBar().addMenu("分析")
        interstitial_action = QAction("间隙分析…", self)
        interstitial_action.setShortcut("Ctrl+I")
        interstitial_action.triggered.connect(self.show_interstitial_analysis)
        analysis_menu.addAction(interstitial_action)

    def _build_view_toolbar(self) -> None:
        toolbar = self.addToolBar("视图控制")
        toolbar.setMovable(False)

        zoom_in = QAction("＋ 放大", self)
        zoom_in.setShortcut("Ctrl+=")
        zoom_in.setToolTip("放大视图")
        zoom_in.triggered.connect(self.viewer_zoom_in)
        toolbar.addAction(zoom_in)

        zoom_out = QAction("－ 缩小", self)
        zoom_out.setShortcut("Ctrl+-")
        zoom_out.setToolTip("缩小视图")
        zoom_out.triggered.connect(self.viewer_zoom_out)
        toolbar.addAction(zoom_out)

        reset_view = QAction("回到初始", self)
        reset_view.setShortcut("Home")
        reset_view.setToolTip("恢复载入结构时的等轴视角和缩放")
        reset_view.triggered.connect(self.viewer_reset_camera)
        toolbar.addAction(reset_view)

        toolbar.addSeparator()
        interstitials = QAction("间隙分析", self)
        interstitials.setToolTip("寻找并显示晶胞中的间隙配位多面体")
        interstitials.triggered.connect(self.show_interstitial_analysis)
        toolbar.addAction(interstitials)

    def _build_interstitial_dock(self) -> None:
        self.interstitial_dock = QDockWidget("间隙分析", self)
        self.interstitial_dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
        )
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)

        radii_group = QGroupBox("障碍原子与分析半径")
        radii_layout = QVBoxLayout(radii_group)
        self.analysis_element_tree = QTreeWidget()
        self.analysis_element_tree.setHeaderLabels(["参与", "元素", "半径/Å"])
        self.analysis_element_tree.setMaximumHeight(170)
        radii_layout.addWidget(self.analysis_element_tree)
        note = QLabel(
            "分析半径独立于左侧显示球半径；部分占位位点当前按存在处理。"
        )
        note.setWordWrap(True)
        radii_layout.addWidget(note)
        layout.addWidget(radii_group)

        self.run_interstitial_button = QPushButton("分析当前 CIF 晶胞")
        self.run_interstitial_button.clicked.connect(self.run_interstitial_analysis)
        layout.addWidget(self.run_interstitial_button)

        result_group = QGroupBox("间隙类型与位点")
        result_layout = QVBoxLayout(result_group)
        self.interstitial_tree = QTreeWidget()
        self.interstitial_tree.setHeaderLabels(
            ["类型/位置/位点", "数量/CN", "自由半径", "中心坐标"]
        )
        self.interstitial_tree.itemClicked.connect(self.interstitial_item_clicked)
        self.interstitial_tree.itemChanged.connect(self.interstitial_item_visibility_changed)
        self.interstitial_tree.setMinimumHeight(330)
        result_layout.addWidget(self.interstitial_tree)
        result_group.setMinimumHeight(380)

        display_group = QGroupBox("多面体显示")
        display_layout = QFormLayout(display_group)
        self.interstitial_opacity_slider = QSlider(Qt.Horizontal)
        self.interstitial_opacity_slider.setRange(5, 90)
        self.interstitial_opacity_slider.setValue(32)
        self.interstitial_opacity_slider.valueChanged.connect(
            lambda value: self.viewer.set_interstitial_opacity(value / 100.0)
        )
        display_layout.addRow("透明度", self.interstitial_opacity_slider)
        self.interstitial_centers_checkbox = QCheckBox("显示间隙中心")
        self.interstitial_centers_checkbox.setChecked(True)
        self.interstitial_centers_checkbox.toggled.connect(
            self.viewer.set_show_interstitial_centers
        )
        display_layout.addRow(self.interstitial_centers_checkbox)
        self.interstitial_spheres_checkbox = QCheckBox("显示最大自由半径球")
        self.interstitial_spheres_checkbox.toggled.connect(
            self.viewer.set_show_interstitial_free_spheres
        )
        display_layout.addRow(self.interstitial_spheres_checkbox)

        info_group = QGroupBox("选中间隙")
        info_form = QFormLayout(info_group)
        self.interstitial_labels: dict[str, QLabel] = {}
        for key, title in [
            ("id", "编号"),
            ("kind", "类型"),
            ("coordination", "配位"),
            ("radius", "自由半径"),
            ("fractional", "分数坐标"),
            ("distortion", "畸变分数"),
        ]:
            label = QLabel("—")
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.interstitial_labels[key] = label
            info_form.addRow(title, label)

        clear_button = QPushButton("清除间隙显示")
        clear_button.clicked.connect(self.clear_interstitial_analysis)

        lower_panel = QWidget()
        lower_layout = QVBoxLayout(lower_panel)
        lower_layout.setContentsMargins(0, 0, 0, 0)
        lower_layout.addWidget(display_group)
        lower_layout.addWidget(info_group)
        lower_layout.addWidget(clear_button)
        lower_layout.addStretch(1)

        self.interstitial_vertical_splitter = QSplitter(Qt.Vertical)
        self.interstitial_vertical_splitter.setChildrenCollapsible(False)
        self.interstitial_vertical_splitter.addWidget(result_group)
        self.interstitial_vertical_splitter.addWidget(lower_panel)
        self.interstitial_vertical_splitter.setSizes([470, 390])
        self.interstitial_vertical_splitter.setMinimumHeight(720)
        layout.addWidget(self.interstitial_vertical_splitter)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(content)
        self.interstitial_scroll_area = scroll

        collapse_button = QPushButton("▼ 收起到下方")
        collapse_button.setToolTip("隐藏间隙分析面板，在窗口底部保留展开按钮")
        collapse_button.clicked.connect(self.interstitial_dock.hide)

        dock_shell = QWidget()
        dock_layout = QVBoxLayout(dock_shell)
        dock_layout.setContentsMargins(0, 0, 0, 6)
        dock_layout.setSpacing(4)
        dock_layout.addWidget(scroll, 1)
        dock_layout.addWidget(collapse_button)
        self.interstitial_dock.setWidget(dock_shell)
        self.addDockWidget(Qt.RightDockWidgetArea, self.interstitial_dock)
        self.interstitial_dock.setMinimumWidth(400)
        self.interstitial_dock.resize(460, 760)

        self.interstitial_restore_button = QPushButton("▲ 展开间隙分析")
        self.interstitial_restore_button.setToolTip("重新显示右侧间隙分析面板")
        self.interstitial_restore_button.clicked.connect(
            self.show_interstitial_analysis
        )
        self.statusBar().addPermanentWidget(self.interstitial_restore_button)
        self.interstitial_dock.visibilityChanged.connect(
            lambda visible: self.interstitial_restore_button.setVisible(not visible)
        )
        self.interstitial_dock.hide()
        self.interstitial_restore_button.show()

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        self.viewer = CrystalViewer(self)
        self.viewer.on_atom_picked = self.show_atom_info
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self.viewer)
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([260, 850, 290])
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        open_button = QPushButton("打开 CIF")
        open_button.clicked.connect(self.open_cif)
        layout.addWidget(open_button)

        element_group = QGroupBox("元素（勾选显示，点击高亮）")
        element_layout = QVBoxLayout(element_group)
        self.element_tree = QTreeWidget()
        self.element_tree.setHeaderLabels(["元素", "原子数", "半径/Å"])
        self.element_tree.itemChanged.connect(self.element_visibility_changed)
        self.element_tree.itemClicked.connect(self.element_clicked)
        element_layout.addWidget(self.element_tree)
        element_buttons = QHBoxLayout()
        clear_highlight = QPushButton("取消高亮")
        clear_highlight.clicked.connect(lambda: self.viewer.highlight_element(None))
        reset_radii = QPushButton("恢复参考半径")
        reset_radii.clicked.connect(self.reset_element_radii)
        element_buttons.addWidget(clear_highlight)
        element_buttons.addWidget(reset_radii)
        element_layout.addLayout(element_buttons)
        layout.addWidget(element_group, 1)

        style_group = QGroupBox("显示样式")
        style_layout = QVBoxLayout(style_group)
        self.bonds_checkbox = QCheckBox("球棍模型（显示化学键）")
        self.bonds_checkbox.setChecked(True)
        self.bonds_checkbox.toggled.connect(self.viewer.set_show_bonds)
        style_layout.addWidget(self.bonds_checkbox)
        self.boundary_checkbox = QCheckBox("补齐晶胞边界与周期成键")
        self.boundary_checkbox.setChecked(True)
        self.boundary_checkbox.toggled.connect(self.viewer.set_complete_boundary)
        style_layout.addWidget(self.boundary_checkbox)
        pair_layout = QHBoxLayout()
        pair_layout.addWidget(QLabel("配位中心"))
        self.center_combo = QComboBox()
        self.center_combo.currentTextChanged.connect(self._bond_settings_changed)
        pair_layout.addWidget(self.center_combo)
        pair_layout.addWidget(QLabel("配位原子"))
        self.ligand_combo = QComboBox()
        self.ligand_combo.currentTextChanged.connect(self._bond_settings_changed)
        pair_layout.addWidget(self.ligand_combo)
        style_layout.addLayout(pair_layout)
        cutoff_layout = QHBoxLayout()
        cutoff_layout.addWidget(QLabel("最大键长"))
        self.bond_cutoff_spin = QDoubleSpinBox()
        self.bond_cutoff_spin.setRange(0.50, 5.00)
        self.bond_cutoff_spin.setSingleStep(0.05)
        self.bond_cutoff_spin.setDecimals(2)
        self.bond_cutoff_spin.setValue(2.40)
        self.bond_cutoff_spin.setSuffix(" Å")
        self.bond_cutoff_spin.valueChanged.connect(self._bond_settings_changed)
        cutoff_layout.addWidget(self.bond_cutoff_spin)
        style_layout.addLayout(cutoff_layout)
        layout.addWidget(style_group)

        supercell_group = QGroupBox("超晶胞")
        supercell_form = QFormLayout(supercell_group)
        self.spin_a = self._supercell_spin()
        self.spin_b = self._supercell_spin()
        self.spin_c = self._supercell_spin()
        supercell_form.addRow("沿 a", self.spin_a)
        supercell_form.addRow("沿 b", self.spin_b)
        supercell_form.addRow("沿 c", self.spin_c)
        buttons = QHBoxLayout()
        apply_button = QPushButton("应用")
        apply_button.clicked.connect(self.apply_supercell)
        reset_button = QPushButton("恢复")
        reset_button.clicked.connect(self.reset_supercell)
        buttons.addWidget(apply_button)
        buttons.addWidget(reset_button)
        supercell_form.addRow(buttons)
        layout.addWidget(supercell_group)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        structure_group = QGroupBox("结构信息")
        form = QFormLayout(structure_group)
        self.info_labels: dict[str, QLabel] = {}
        for key, title in [
            ("file", "文件"), ("formula", "组成"), ("spacegroup", "空间群"),
            ("sites", "晶胞位点"), ("ordered", "有序性"), ("lattice", "晶格参数"),
            ("angles", "晶格角"), ("volume", "体积"),
        ]:
            label = QLabel("—")
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.info_labels[key] = label
            form.addRow(title, label)
        layout.addWidget(structure_group)

        atom_group = QGroupBox("选中原子")
        atom_form = QFormLayout(atom_group)
        self.atom_labels: dict[str, QLabel] = {}
        for key, title in [
            ("index", "序号"), ("element", "元素"), ("species", "物种"),
            ("occupancy", "占位率"), ("fractional", "分数坐标"),
            ("cartesian", "笛卡尔坐标"),
        ]:
            label = QLabel("—")
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.atom_labels[key] = label
            atom_form.addRow(title, label)
        layout.addWidget(atom_group)
        layout.addStretch(1)
        return panel

    @staticmethod
    def _supercell_spin() -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(1, 8)
        spin.setValue(1)
        return spin

    def open_cif(self) -> None:
        start = self.settings.value("last_directory", str(Path.home()))
        filename, _ = QFileDialog.getOpenFileName(self, "打开 CIF", start, "CIF 文件 (*.cif);;所有文件 (*)")
        if filename:
            self.load_cif(filename)

    def load_cif(self, filename: str) -> None:
        try:
            document = CrystalDocument.from_cif(filename)
        except Exception as exc:
            QMessageBox.critical(self, "无法打开 CIF", str(exc))
            return
        self.document = document
        self.settings.setValue("last_directory", str(Path(filename).parent))
        self.viewer.set_document(document)
        self._populate_elements()
        self._show_summary()
        self.spin_a.setValue(1)
        self.spin_b.setValue(1)
        self.spin_c.setValue(1)
        warning_count = len(document.parser_warnings)
        suffix = f"；解析器给出 {warning_count} 条提示" if warning_count else ""
        self.statusBar().showMessage(f"已载入 {Path(filename).name}，{len(document.display)} 个位点{suffix}")

    def _populate_elements(self) -> None:
        assert self.document is not None
        counts: dict[str, int] = {}
        for site in self.document.display:
            element = site_element(site)
            counts[element] = counts.get(element, 0) + 1
        self.element_tree.blockSignals(True)
        self.element_tree.clear()
        self.radius_spins: dict[str, QDoubleSpinBox] = {}
        for element in sorted(counts):
            radius = self.viewer.element_radii.get(element, 0.70)
            item = QTreeWidgetItem([element, str(counts[element]), ""])
            item.setData(0, Qt.UserRole, element)
            item.setCheckState(0, Qt.Checked)
            item.setForeground(0, QColor(color_for(element)))
            self.element_tree.addTopLevelItem(item)
            radius_spin = QDoubleSpinBox()
            radius_spin.setRange(0.05, 3.00)
            radius_spin.setDecimals(2)
            radius_spin.setSingleStep(0.05)
            radius_spin.setValue(radius)
            radius_spin.setSuffix(" Å")
            radius_spin.setToolTip(f"{element} 的显示球半径；不影响坐标和成键判断")
            radius_spin.valueChanged.connect(
                lambda value, symbol=element: self.viewer.set_element_radius(symbol, value)
            )
            self.radius_spins[element] = radius_spin
            self.element_tree.setItemWidget(item, 2, radius_spin)
        self.element_tree.resizeColumnToContents(0)
        self.element_tree.resizeColumnToContents(1)
        elements = sorted(counts)
        self.center_combo.blockSignals(True)
        self.ligand_combo.blockSignals(True)
        self.center_combo.clear()
        self.ligand_combo.clear()
        self.center_combo.addItems(elements)
        self.ligand_combo.addItems(elements)
        center = "Nb" if "Nb" in elements else elements[0]
        ligand = "O" if "O" in elements else elements[-1]
        self.center_combo.setCurrentText(center)
        self.ligand_combo.setCurrentText(ligand)
        self.center_combo.blockSignals(False)
        self.ligand_combo.blockSignals(False)
        self.viewer.set_bond_settings(center, ligand, self.bond_cutoff_spin.value())
        self.element_tree.blockSignals(False)
        self._populate_analysis_elements(elements)
        self.interstitial_result = None
        self.interstitial_tree.clear()
        self._clear_interstitial_labels()

    def _populate_analysis_elements(self, elements: list[str]) -> None:
        self.analysis_element_tree.clear()
        self.analysis_radius_spins = {}
        for element in elements:
            item = QTreeWidgetItem(["", element, ""])
            item.setCheckState(0, Qt.Checked)
            item.setForeground(1, QColor(color_for(element)))
            self.analysis_element_tree.addTopLevelItem(item)
            spin = QDoubleSpinBox()
            spin.setRange(0.00, 3.00)
            spin.setDecimals(2)
            spin.setSingleStep(0.05)
            spin.setSuffix(" Å")
            spin.setValue(REFERENCE_ANALYSIS_RADII.get(element, 0.90))
            spin.setToolTip(f"{element} 的几何障碍半径")
            self.analysis_radius_spins[element] = spin
            self.analysis_element_tree.setItemWidget(item, 2, spin)
        self.analysis_element_tree.resizeColumnToContents(0)
        self.analysis_element_tree.resizeColumnToContents(1)

    def _show_summary(self) -> None:
        assert self.document is not None
        info = self.document.summary()
        self.info_labels["file"].setText(info.source.name)
        self.info_labels["formula"].setText(f"{info.formula}  ({info.reduced_formula})")
        number = f"No. {info.space_group_number}" if info.space_group_number else "编号未知"
        self.info_labels["spacegroup"].setText(f"{info.space_group} · {number}")
        self.info_labels["sites"].setText(str(info.site_count))
        self.info_labels["ordered"].setText("完全占位" if info.is_ordered else "含部分占位/无序")
        self.info_labels["lattice"].setText(f"a {info.a:.4f} Å\nb {info.b:.4f} Å\nc {info.c:.4f} Å")
        self.info_labels["angles"].setText(f"α {info.alpha:.3f}°\nβ {info.beta:.3f}°\nγ {info.gamma:.3f}°")
        self.info_labels["volume"].setText(f"{info.volume:.3f} Å³")

    def element_visibility_changed(self, item: QTreeWidgetItem, column: int) -> None:
        element = item.data(0, Qt.UserRole)
        if column == 0:
            self.viewer.set_element_visible(element, item.checkState(0) == Qt.Checked)

    def element_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        self.viewer.highlight_element(item.data(0, Qt.UserRole))

    def reset_element_radii(self) -> None:
        for element, spin in self.radius_spins.items():
            value = radius_for(element)
            self.viewer.element_radii[element] = value
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.viewer.render_structure(reset_camera=False)
        self.statusBar().showMessage("已恢复元素参考显示半径")

    def _bond_settings_changed(self, _value=None) -> None:
        if not self.center_combo.currentText() or not self.ligand_combo.currentText():
            return
        self.viewer.set_bond_settings(
            self.center_combo.currentText(),
            self.ligand_combo.currentText(),
            self.bond_cutoff_spin.value(),
        )

    def show_interstitial_analysis(self) -> None:
        self.interstitial_dock.show()
        self.interstitial_dock.raise_()

    def run_interstitial_analysis(self) -> None:
        if self.document is None:
            QMessageBox.information(self, "尚无结构", "请先打开一个 CIF 文件。")
            return
        obstacle_elements = set()
        for row in range(self.analysis_element_tree.topLevelItemCount()):
            item = self.analysis_element_tree.topLevelItem(row)
            if item.checkState(0) == Qt.Checked:
                obstacle_elements.add(item.text(1))
        if not obstacle_elements:
            QMessageBox.warning(self, "没有障碍原子", "请至少选择一种参与分析的元素。")
            return
        radii = {
            element: spin.value()
            for element, spin in self.analysis_radius_spins.items()
        }
        self.run_interstitial_button.setEnabled(False)
        self.statusBar().showMessage("正在分析间隙…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = InterstitialAnalyzer().analyze(
                self.document.original,
                obstacle_elements=obstacle_elements,
                radii=radii,
            )
        except Exception as exc:
            QMessageBox.critical(self, "间隙分析失败", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.run_interstitial_button.setEnabled(True)

        self.interstitial_result = result
        self._populate_interstitial_results(result)
        if result.kinds:
            first_kind = result.kinds[0]
            self.viewer.set_interstitial_result(result, first_kind)
            first_item = self.interstitial_tree.topLevelItem(0)
            if first_item is not None:
                self.interstitial_tree.setCurrentItem(first_item)
        else:
            self.viewer.clear_interstitials()
        self.statusBar().showMessage(
            f"间隙分析完成：{len(result.sites)} 个候选位点，{len(result.kinds)} 种几何类型"
        )

    def _populate_interstitial_results(self, result: InterstitialResult) -> None:
        self.interstitial_tree.blockSignals(True)
        self.interstitial_tree.clear()
        for kind in result.kinds:
            sites = result.sites_of_kind(kind)
            radii = [site.free_radius for site in sites]
            radius_text = (
                f"{radii[0]:.3f} Å"
                if max(radii) - min(radii) < 1e-6
                else f"{min(radii):.3f}–{max(radii):.3f} Å"
            )
            parent = QTreeWidgetItem([kind, str(len(sites)), radius_text])
            parent.setData(0, Qt.UserRole, ("kind", kind))
            parent.setFlags(
                parent.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate
            )
            parent.setCheckState(0, Qt.Checked)
            self.interstitial_tree.addTopLevelItem(parent)
            position_groups: dict[str, list] = {}
            for site in sites:
                position_groups.setdefault(
                    fractional_position_category(site.frac_coords), []
                ).append(site)
            for position in ("晶胞内部", "晶胞面上", "晶胞棱上", "晶胞角点"):
                grouped_sites = position_groups.get(position, [])
                if not grouped_sites:
                    continue
                group = QTreeWidgetItem([position, str(len(grouped_sites)), "", ""])
                group.setData(0, Qt.UserRole, ("position", kind, position))
                group.setFlags(
                    group.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate
                )
                group.setCheckState(0, Qt.Checked)
                parent.addChild(group)
                for site in grouped_sites:
                    coordinate_text = "(" + ", ".join(
                        f"{value:.4f}" for value in site.frac_coords
                    ) + ")"
                    child = QTreeWidgetItem([
                        f"I{site.site_id}",
                        f"CN {site.coordination_number}",
                        f"{site.free_radius:.3f} Å",
                        coordinate_text,
                    ])
                    child.setData(0, Qt.UserRole, ("site", site.site_id, kind))
                    child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                    child.setCheckState(0, Qt.Checked)
                    child.setToolTip(0, "分数坐标：" + coordinate_text)
                    group.addChild(child)
                group.setExpanded(True)
            parent.setExpanded(True)
        self.interstitial_tree.blockSignals(False)
        self.interstitial_tree.resizeColumnToContents(0)
        self.interstitial_tree.resizeColumnToContents(1)

    def _checked_interstitial_site_ids(self, kind: str) -> set[int]:
        checked: set[int] = set()
        for row in range(self.interstitial_tree.topLevelItemCount()):
            parent = self.interstitial_tree.topLevelItem(row)
            data = parent.data(0, Qt.UserRole)
            if not data or data[0] != "kind" or data[1] != kind:
                continue
            for group_index in range(parent.childCount()):
                group = parent.child(group_index)
                for site_index in range(group.childCount()):
                    child = group.child(site_index)
                    child_data = child.data(0, Qt.UserRole)
                    if child_data and child_data[0] == "site" and child.checkState(0) == Qt.Checked:
                        checked.add(int(child_data[1]))
            break
        return checked

    def interstitial_item_visibility_changed(
        self,
        item: QTreeWidgetItem,
        _column: int,
    ) -> None:
        data = item.data(0, Qt.UserRole)
        if not data or self.interstitial_result is None:
            return
        if data[0] == "kind":
            kind = data[1]
        elif data[0] == "position":
            kind = data[1]
        elif data[0] == "site":
            kind = data[2]
        else:
            return
        self._pending_interstitial_visibility_kind = kind
        if not self._interstitial_visibility_update_scheduled:
            self._interstitial_visibility_update_scheduled = True
            QTimer.singleShot(0, self._apply_pending_interstitial_visibility)

    def _apply_pending_interstitial_visibility(self) -> None:
        self._interstitial_visibility_update_scheduled = False
        kind = self._pending_interstitial_visibility_kind
        self._pending_interstitial_visibility_kind = None
        if kind is None or self.interstitial_result is None:
            return
        self.viewer.set_visible_interstitial_sites(
            kind,
            self._checked_interstitial_site_ids(kind),
        )

    def interstitial_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        data = item.data(0, Qt.UserRole)
        if not data or self.interstitial_result is None:
            return
        category, value = data[:2]
        if category == "kind":
            self.viewer.set_visible_interstitial_sites(
                value,
                self._checked_interstitial_site_ids(value),
            )
            self._clear_interstitial_labels()
            return
        if category == "position":
            self.viewer.set_visible_interstitial_sites(
                value,
                self._checked_interstitial_site_ids(value),
            )
            self._clear_interstitial_labels()
            return
        site = next(
            (candidate for candidate in self.interstitial_result.sites if candidate.site_id == value),
            None,
        )
        if site is None:
            return
        self.viewer.select_interstitial(site.site_id)
        self._show_interstitial_info(site)

    def _show_interstitial_info(self, site) -> None:
        self.interstitial_labels["id"].setText(f"I{site.site_id}")
        self.interstitial_labels["kind"].setText(site.kind)
        elements: dict[str, int] = {}
        for neighbor in site.neighbors:
            elements[neighbor.element] = elements.get(neighbor.element, 0) + 1
        environment = " + ".join(f"{element}×{count}" for element, count in sorted(elements.items()))
        self.interstitial_labels["coordination"].setText(
            f"CN {site.coordination_number}；{environment}"
        )
        self.interstitial_labels["radius"].setText(f"{site.free_radius:.4f} Å")
        self.interstitial_labels["fractional"].setText(
            "\n".join(
                f"{axis}  {value:.6f}"
                for axis, value in zip(("x", "y", "z"), site.frac_coords)
            )
        )
        self.interstitial_labels["distortion"].setText(f"{site.distortion:.4f}")

    def _clear_interstitial_labels(self) -> None:
        for label in getattr(self, "interstitial_labels", {}).values():
            label.setText("—")

    def clear_interstitial_analysis(self) -> None:
        self.viewer.clear_interstitials()
        self.interstitial_result = None
        self.interstitial_tree.clearSelection()
        self._clear_interstitial_labels()
        self.statusBar().showMessage("已清除间隙显示")

    def apply_supercell(self) -> None:
        if self.document is None:
            return
        repeats = (self.spin_a.value(), self.spin_b.value(), self.spin_c.value())
        estimated = len(self.document.original) * repeats[0] * repeats[1] * repeats[2]
        if estimated > 5000:
            answer = QMessageBox.question(
                self,
                "大型超晶胞",
                f"预计显示 {estimated} 个原子，可能较慢。是否继续？",
            )
            if answer != QMessageBox.Yes:
                return
        self.document.set_supercell(*repeats)
        self.viewer.set_document(self.document)
        self._populate_elements()
        self.statusBar().showMessage(f"超晶胞 {repeats[0]} × {repeats[1]} × {repeats[2]}，{len(self.document.display)} 个位点")

    def reset_supercell(self) -> None:
        if self.document is None:
            return
        self.spin_a.setValue(1)
        self.spin_b.setValue(1)
        self.spin_c.setValue(1)
        self.apply_supercell()

    def show_atom_info(self, index: int, offset: tuple[int, int, int] = (0, 0, 0)) -> None:
        if self.document is None or not 0 <= index < len(self.document.display):
            return
        site = self.document.display[index]
        image = "" if offset == (0, 0, 0) else f"  周期像 {offset}"
        self.atom_labels["index"].setText(f"{index + 1}{image}")
        self.atom_labels["element"].setText(site_element(site))
        self.atom_labels["species"].setText(str(site.species))
        self.atom_labels["occupancy"].setText(f"{site_occupancy(site):.4f}")
        fractional = site.frac_coords + offset
        cartesian = fractional @ self.document.display.lattice.matrix
        self.atom_labels["fractional"].setText(
            "\n".join(
                f"{axis}  {value:.6f}"
                for axis, value in zip(("x", "y", "z"), fractional)
            )
        )
        self.atom_labels["cartesian"].setText(
            "\n".join(
                f"{axis}  {value:.4f} Å"
                for axis, value in zip(("x", "y", "z"), cartesian)
            )
        )

    def save_screenshot(self) -> None:
        if self.document is None:
            QMessageBox.information(self, "尚无结构", "请先打开一个 CIF 文件。")
            return
        suggested = str(self.document.source.with_suffix(".png"))
        filename, _ = QFileDialog.getSaveFileName(self, "保存视图", suggested, "PNG 图片 (*.png)")
        if filename:
            self.viewer.screenshot(filename)
            self.statusBar().showMessage(f"图片已保存到 {filename}")

    def viewer_reset_camera(self) -> None:
        self.viewer.reset_to_home_view()

    def viewer_zoom_in(self) -> None:
        self.viewer.zoom_by(1.20)

    def viewer_zoom_out(self) -> None:
        self.viewer.zoom_by(1.0 / 1.20)

    def closeEvent(self, event) -> None:
        self.viewer.close()
        super().closeEvent(event)
