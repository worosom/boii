$(function() {
	example1_chan1 = new Audio('mp3/gallery_clean.mp3');
	example1_chan2 = new Audio('mp3/gallery_boii.mp3');
	example1_chan1.volume = .5;
	example1_chan2.volume = .5;

	example2_chan1 = new Audio('mp3/jam_clean.mp3');
	example2_chan2 = new Audio('mp3/jam_boii.mp3');
	example2_chan1.volume = .5;
	example2_chan2.volume = .5;

	example3_chan1 = new Audio('mp3/last_day_clean.mp3');
	example3_chan2 = new Audio('mp3/last_day_boii.mp3');
	example3_chan1.volume = .5;
	example3_chan2.volume = .5;

	/**************************************/

    $(".example1").knob({
    width: 150,
    height: 150,
    displayInput: false,
    displayPrevious: false,
    cursor: 10,
    thickness: .25,
    min: 0,
    max: 1,
    step: 1/512.,
    stopper: false,
    angleArc: 250,
    angleOffset:-125,
    fgColor:'#2255EE',
    change: function (value) {
    	example1_chan1.volume = 1.-value;
    	example1_chan2.volume = value;
    },
    'draw': function () {
        if ($(this.i).siblings(".play").length > 0) return
        	var pos1 = parseInt($(this.i).css("marginLeft").replace('px', ''));
        	var pos2 = parseInt($(this.i).css("marginTop").replace('px', ''));
    	    var $elem = $("<span></span>").addClass("btn_play");
    	    $elem.addClass("example1")
	        $elem.insertAfter(this.i).css({
        	    position: "absolute",
    	        marginLeft: (pos1-100) + "px",
	            marginTop: (pos2 + 45) + "px"
        	});
    	}
	});
	$("body").on("click", ".example1", function () {
		if (example1_chan1.paused && example1_chan1.readyState && example1_chan2.readyState){
			if (example1_chan1.currentTime != example1_chan2.currentTime ) {
				example1_chan2.currentTime = example1_chan1.currentTime;
			}
    		example1_chan1.play();
    		example1_chan2.play();
    		$(".example1").css({"background-position": "0 100%"});
    	}
    	else {
    		example1_chan1.pause();
    		example1_chan2.pause();
    		$(".example1").css({"background-position": "0 0"});
    	}
	});

	/**************************************/

	$(".example2").knob({
    width: 150,
    height: 150,
    displayInput: false,
    displayPrevious: false,
    cursor: 10,
    thickness: .25,
    min: 0,
    max: 1,
    step: 1/512.,
    stopper: false,
    angleArc: 250,
    angleOffset:-125,
    fgColor:'#2255EE',
    change: function (value) {
    	example2_chan1.volume = 1.-value;
    	example2_chan2.volume = value;
    },
    'draw': function () {
        if ($(this.i).siblings(".play").length > 0) return
        	var pos1 = parseInt($(this.i).css("marginLeft").replace('px', ''));
        	var pos2 = parseInt($(this.i).css("marginTop").replace('px', ''));
    	    var $elem = $("<span></span>").addClass("btn_play");
    	    $elem.addClass("example2_button")
	        $elem.insertAfter(this.i).css({
        	    position: "absolute",
    	        marginLeft: (pos1-100) + "px",
	            marginTop: (pos2 + 45) + "px"
        	});
    	}
	});
	$("body").on("click", ".example2_button", function () {
		if (example2_chan1.paused && example2_chan1.readyState && example2_chan2.readyState){
			if (example2_chan1.currentTime != example2_chan2.currentTime ) {
				example2_chan2.currentTime = example2_chan1.currentTime;
			}
    		example2_chan1.play();
    		example2_chan2.play();
    		$(".example2_button").css({"background-position": "0 100%"});
    	}
    	else {
    		example2_chan1.pause();
    		example2_chan2.pause();
    		$(".example2_button").css({"background-position": "0 0"});
    	}
	});

	/**************************************/

	$(".example3").knob({
    width: 150,
    height: 150,
    displayInput: false,
    displayPrevious: false,
    cursor: 10,
    thickness: .25,
    min: 0,
    max: 1,
    step: 1/512.,
    stopper: false,
    angleArc: 250,
    angleOffset:-125,
    fgColor:'#2255EE',
    change: function (value) {
    	example3_chan1.volume = 1.-value;
    	example3_chan2.volume = value;
    },
    'draw': function () {
        if ($(this.i).siblings(".play").length > 0) return
        	var pos1 = parseInt($(this.i).css("marginLeft").replace('px', ''));
        	var pos2 = parseInt($(this.i).css("marginTop").replace('px', ''));
    	    var $elem = $("<span></span>").addClass("btn_play");
    	    $elem.addClass("example3_button")
	        $elem.insertAfter(this.i).css({
        	    position: "absolute",
    	        marginLeft: (pos1-100) + "px",
	            marginTop: (pos2 + 45) + "px"
        	});
    	}
	});
	$("body").on("click", ".example3_button", function () {
		if (example3_chan1.paused && example3_chan1.readyState && example3_chan2.readyState){
			if (example3_chan1.currentTime != example3_chan2.currentTime ) {
				example3_chan2.currentTime = example3_chan1.currentTime;
			}
    		example3_chan1.play();
    		example3_chan2.play();
    		$(".example3_button").css({"background-position": "0 100%"});
    	}
    	else {
    		example3_chan1.pause();
    		example3_chan2.pause();
    		$(".example3_button").css({"background-position": "0 0"});
    	}
	});

    
    $(window).bind('scroll', chk_scroll);
    $("#backbutton").animate({width: "100%"}).data("scaled", false);
    function chk_scroll(e) {
        var elem = $(e.currentTarget);
        if ($("#1").offset().top - elem.scrollTop() < 0) {
            if(!$("#backbutton").data("scaled")){
                $("#backbutton").animate({width: "50%"});
            }
            $("#backbutton").data("scaled", true);
        } 
        else if ($("#backbutton").data("scaled")){
            $("#backbutton").data("scaled", false).animate({width: "100%"}).promise();
        }
    }

	$("a").click(function (event) {
		if(location.hostname == "mbp" && $(this).attr('href').length > 2){
			event.preventDefault()
			var $elem = $('<div></div>');
			var $p = $("<p>Please visit<br><a>https://worosom.github.io/boii</a><br>to follow this link.</p>");
			$p.css({
				"font-size": "1.5rem",
				"position": "relative",
				"top": "50%",
				"marginTop": "-1.5rem"
			});
			$p.appendTo($elem);
			$elem.css({
				"position": "fixed",
				"top": "25%",
				"left": "0",
				"width": "100vw",
				"height": "25vh",
				"background-color": "#A5C0CC",
				"text-align": "center",
				"display": "none"});
			$elem.addClass("banner")
			$elem.appendTo("body");
			$elem.fadeIn(500).delay(3000).fadeOut("slow").promise().then(function(){
					$(".banner").remove();
			});
		}
	});
});
