from deck import blank, slide, add_element, element


def example():
    d = blank("光与折射")
    p = d["slides"]["p1"]
    p["title"] = "光的折射"

    def put(page, key, kind, **values):
        add_element(page, element(kind, **values), key)

    put(p, "rule", "rect", name="标题线", x=58, y=74, w=54, h=4, fill="#CB663E")
    put(
        p,
        "topic",
        "text",
        name="章节",
        text="高中物理    /    光学",
        x=58,
        y=94,
        w=385,
        h=36,
        size=16,
        color="#97543F",
        bold=True,
    )
    put(
        p,
        "title",
        "text",
        name="标题",
        text="光的折射",
        x=53,
        y=139,
        w=433,
        h=111,
        size=78,
        font="serif",
        bold=True,
    )
    put(
        p,
        "lead",
        "text",
        name="导语",
        text="光穿过两种介质的界面时，\n传播方向会怎样改变？",
        x=58,
        y=290,
        w=400,
        h=115,
        size=24,
        color="#465954",
    )
    put(p, "water", "rect", name="水", x=490, y=260, w=410, h=166, fill="#C7E3DB")
    put(p, "surface", "line", name="介质界面", x=490, y=260, w=410, h=2, fill="#639589")
    put(p, "normal", "line", name="法线", x=705, y=110, w=2, h=290, fill="#A3B2A9")
    put(
        p,
        "incoming",
        "arrow",
        name="入射光线",
        x=546,
        y=188,
        w=188,
        h=6,
        rotation=46,
        fill="#C35D39",
    )
    put(
        p,
        "outgoing",
        "arrow",
        name="折射光线",
        x=672,
        y=318,
        w=145,
        h=6,
        rotation=57,
        fill="#227566",
    )
    put(
        p,
        "airlabel",
        "text",
        name="空气标注",
        text="空气",
        x=829,
        y=221,
        w=60,
        h=30,
        size=17,
        color="#67756D",
    )
    put(
        p,
        "waterlabel",
        "text",
        name="水标注",
        text="水",
        x=837,
        y=282,
        w=45,
        h=30,
        size=17,
        color="#416D63",
    )
    put(
        p,
        "caption",
        "text",
        name="图注",
        text="图 1　光从空气斜射入水",
        x=507,
        y=449,
        w=390,
        h=24,
        size=14,
        color="#68766D",
        align="center",
    )
    for i, (title, txt) in enumerate(
        [
            (
                "折射定律",
                "折射光线、入射光线与法线在同一平面内。\n\nn₁ sin θ₁ = n₂ sin θ₂",
            ),
            (
                "课堂思考",
                "如果让光沿折射光线的方向反向传播，\n它将沿怎样的路径离开水面？",
            ),
        ],
        2,
    ):
        page = slide(title)
        put(
            page,
            "heading",
            "text",
            name="标题",
            text=title,
            x=70,
            y=78,
            w=760,
            h=100,
            size=58,
            font="serif",
            bold=True,
        )
        put(
            page,
            "body",
            "text",
            name="正文",
            text=txt,
            x=75,
            y=228,
            w=810,
            h=215,
            size=30,
        )
        d["slides"][f"p{i}"] = page
        d["order"].append(f"p{i}")
    return d
